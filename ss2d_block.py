import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,  
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="xavier", 
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model

        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)  # d_inner = 2 * d_model
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.norm_conv = nn.BatchNorm2d(self.d_inner)  
        self.act = nn.LeakyReLU(0.1, inplace=True)  

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj  

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs  

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self.fusion_conv = nn.Conv1d(
            in_channels=4 * self.d_inner,  
            out_channels=self.d_inner,    
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="xavier", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        if dt_init == "xavier":
            nn.init.xavier_uniform_(dt_proj.weight)
        elif dt_init == "random":
            dt_init_std = dt_rank**-0.5 * dt_scale
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([
            x.view(B, -1, L), 
            torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)
        ], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (B, K=4, d_inner, L)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)  
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)  

        xs = xs.float().view(B, -1, L) # (B, K * d_inner, L)
        dts = dts.contiguous().float().view(B, -1, L) # (B, K * d_inner, L)
        Bs = Bs.float().view(B, K, -1, L) # (B, K, d_state, L)
        Cs = Cs.float().view(B, K, -1, L) # (B, K, d_state, L)
        Ds = self.Ds.float().view(-1) # (K * d_state)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (K * d_state, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (K * d_state)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float32

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        y1 = out_y[:, 0]      # (B, d_inner, L)
        y2 = inv_y[:, 0]      # (B, d_inner, L)
        y3 = wh_y             # (B, d_inner, L)
        y4 = invwh_y          # (B, d_inner, L)
        y = torch.cat([y1, y2, y3, y4], dim=1)  
        y = self.fusion_conv(y)  

        return y 

    def forward(self, x, H, W, relative_pos=None):
        B, N, C = x.shape
        x_residual = x.clone()  


        x = x.permute(0, 2, 1).reshape(B, H, W, C)
        # print(f"x reshaped: {x.shape}")  # (B, H, W, C)


        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (B, H, W, d_inner), (B, H, W, d_inner)
        # print(f"x after in_proj: {x.shape}, z shape: {z.shape}")


        x = x.permute(0, 3, 1, 2).contiguous()  # (B, d_inner, H, W)
        # print(f"x permuted: {x.shape}")


        x = self.conv2d(x)  # (B, d_inner, H, W)
        x = self.norm_conv(x)  # (B, d_inner, H, W)
        x = self.act(x)  
        # print(f"x after conv, norm, act: {x.shape}")


        y = self.forward_core(x)  # y: (B, d_inner, L)
        # print(f"y from forward_core: {y.shape}")


        y = y.view(B, self.d_inner, H, W)  # (B, d_inner, H, W)
        # print(f"y after view: {y.shape}")


        y = self.out_norm(y.permute(0, 2, 3, 1))  # (B, H, W, d_inner)
        # print(f"y after out_norm: {y.shape}")


        y = y * F.leaky_relu(z)  # (B, H, W, d_inner)
        # print(f"y after multiplication: {y.shape}")


        out = self.out_proj(y)  # (B, d_model, H, W)
        if self.dropout is not None:
            out = self.dropout(out)
        out = out.reshape(B, N, C)  # (B, N, C)
        # print(f"out after out_proj and reshape: {out.shape}")


        out = out + x_residual  # (B, N, C)
        # print(f"out after residual connection: {out.shape}")

        return out


class SS2D_Block(nn.Module):
    def __init__(self, channels):
        super(SS2D_Block, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.ss2d = SS2D(d_model=channels, dropout=0, d_state=16, expand=2)  
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.norm = nn.BatchNorm2d(channels)
        self.activation = nn.SiLU()  

    def forward(self, x):          
        residual = x 
        x = self.conv1(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  
        x = self.ss2d(x, H, W) 
        x = x.transpose(1, 2).view(B, C, H, W)       
        x = self.conv2(x)   
        x = self.norm(x)    
        x = x + residual  
        x = self.activation(x)
        return x 
