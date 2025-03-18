import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Pscan import pscan
from layers.GDDMLP import GDDMLP
from mamba_ssm import Mamba
# from mamba_ssm import Mamba2
from layers.SelfAttention_Family import AttentionLayer, FullAttention

class CMambaEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.configs = configs
        self.layers = nn.ModuleList([CMambaBlock(configs) for _ in range(configs.e_layers)])

    def forward(self, x):
        # x : [bs * nvars, patch_num, d_model]

        for layer in self.layers:
            x = layer(x)

        x = F.silu(x)

        return x

class CMambaBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.mixer = MambaBlock(configs)
        self.norm = RMSNorm(configs.d_model)

        self.gddmlp = configs.gddmlp
        # if self.gddmlp:
        #     print("Insert GDDMLP")
        #     self.GDDMLP = GDDMLP(configs.c_out, configs.reduction, 
        #                                          configs.avg, configs.max)
        self.conv1 = nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=configs.d_ff, out_channels=configs.d_model, kernel_size=1)
        self.activation = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.configs = configs

    def forward(self, x):
        # x: [B*patch_num, L, d_model]
        # print("CMambaBlock: x.shape", x.shape)
        # output : [B*patch_num, L, d_model]
        output = self.mixer(x)
        # print("CMambaBlock: output.shape", output.shape)
        # if self.gddmlp:
        #     print("GDDMLP==========")
        #     # output : [bs, nvars, patch_num, d_model]
        #     output = self.GDDMLP(output.reshape(-1, self.configs.c_out, 
        #                                                   output.shape[-2], output.shape[-1]))
        #     # output : [bs * nvars, patch_num, d_model]
        #     output = output.reshape(-1, output.shape[-2], output.shape[-1])
        output = self.dropout(output)
        output += x

        o = output.permute(0, 2, 1)        # [B*patch_num, d_model, L]
        y = self.dropout(self.activation(self.conv1(o)))    # [B*patch_num, d_ff, L]
        y = self.dropout(self.conv2(y))     # [B*patch_num, d_model, L]
        y = y.permute(0, 2, 1)              # [B*patch_num, L, d_model]
        return y

class MambaBlock(nn.Module):
    """
    MambaModule, similar to https://arxiv.org/pdf/2402.18959
    """
    def __init__(self, configs):
        super().__init__()

        self.configs = configs

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(configs.d_model, 2 * configs.d_ff, bias=configs.bias)
        
        # projects x to input-dependent Δ, B, C, D
        self.x_proj = nn.Linear(configs.d_ff, configs.dt_rank + 2 * configs.d_state + configs.d_ff, bias=False)

        # projects Δ from dt_rank to d_ff
        self.dt_proj = nn.Linear(configs.dt_rank, configs.d_ff, bias=True)

        # dt initialization
        # dt weights
        dt_init_std = configs.dt_rank**-0.5 * configs.dt_scale
        if configs.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif configs.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        # dt bias
        dt = torch.exp(
            torch.rand(configs.d_ff) * (math.log(configs.dt_max) - math.log(configs.dt_min)) + math.log(configs.dt_min)
        ).clamp(min=configs.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D real initialization
        A = torch.arange(1, configs.d_state + 1, dtype=torch.float32).unsqueeze(0) # shape: [1, configs.d_state]

        self.A_log = nn.Parameter(torch.log(A))

        # projects block output from ED back to D
        self.out_proj = nn.Linear(configs.d_ff, configs.d_model, bias=configs.bias)

    def forward(self, x):
        # x : [B*patch_num, L, d_model]
        # y : [B*patch_num, L, d_model]

        _, L, _ = x.shape
        # print("====================MambaBlock===============================")
        xz = self.in_proj(x) # [B*patch_num, L, 2 * d_ff]
        # 沿最后一个维度分割
        x, z = xz.chunk(2, dim=-1) # [B*patch_num, L, d_ff], [B*patch_num, L, d_ff]
        # x branch
        x = F.silu(x)
        y = self.ssm(x)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) # [bs * nvars, patch_num, d_ff]

        return output
    
    def ssm(self, x):
        # print("=========SSM=========")
        # x : [B*patch_num, L, d_ff]

        # y : [B*patch_num, L, d_ff]
        A = -torch.exp(self.A_log.float()) # [d_ff, d_state]
        # print("A:", A.shape)

        deltaBCD = self.x_proj(x) # [B*patch_num, L, dt_rank + 2 * d_state + d_ff]
        # [B*patch_num, L, dt_rank], [B*patch_num, L, d_state], [B*patch_num, L, d_state], [B*patch_num, L, d_ff]
        delta, B, C, D = torch.split(deltaBCD, [self.configs.dt_rank, self.configs.d_state, self.configs.d_state, self.configs.d_ff], dim=-1)
        delta = F.softplus(self.dt_proj(delta)) # [bs * nvars, patch_num, d_ff]
        # print(f"delta.shape: {delta.shape}, B.shape: {B.shape}, C.shape: {C.shape}, D.shape: {D.shape}")
        # print("x.shape:", x.shape)
        if self.configs.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y
    
    def selective_scan(self, x, delta, A, B, C, D):
        # x : [B*patch_num, L, d_ff]
        # Δ : [B*patch_num, L, d_ff]
        # A : [d_ff, d_state]
        # B : [B*patch_num, L, d_state]
        # C : [B*patch_num, L, d_state]
        # D : [B*patch_num, L, d_ff]

        # y : [B*patch_num, L, d_ff]

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # [B*patch_num, L, d_ff, d_state]
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # [B*patch_num, L, d_ff, d_state]
        # print(f"deltaA.shape: {deltaA.shape}, deltaB.shape: {deltaB.shape}")
        BX = deltaB * (x.unsqueeze(-1)) # [B*patch_num, L, d_ff, d_state]
        # print(f"BX.shape: {BX.shape}")
        hs = pscan(deltaA, BX)
        # [B*patch_num, L, d_ff, d_state] @ [B*patch_num, L, d_state, 1] ->[B*patch_num, L, d_ff]
        y = (hs @ C.unsqueeze(-1)).squeeze(3)
        # print(f"y.shape: {y.shape}")
        y = y + D * x
        # print(f"y.shape: {y.shape}")
        return y
    
    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : [bs * nvars, patch_num, d_ff]
        # Δ : [bs * nvars, patch_num, d_ff]
        # A : [d_ff, d_state]
        # B : [bs * nvars, patch_num, d_state]
        # C : [bs * nvars, patch_num, d_state]
        # D : [bs * nvars, patch_num, d_ff]

        # y : [bs * nvars, patch_num, d_ff]

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # [bs * nvars, patch_num, d_ff, d_state]
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # [bs * nvars, patch_num, d_ff, d_state]

        BX = deltaB * (x.unsqueeze(-1)) # [bs * nvars, patch_num, d_ff, d_state]

        h = torch.zeros(x.size(0), self.configs.d_ff, self.configs.d_state, device=deltaA.device) # (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
            
        hs = torch.stack(hs, dim=1) # [bs * nvars, patch_num, d_ff, d_state]
        # [bs * nvars, patch_num, d_ff, d_state] @ [bs * nvars, patch_num, d_state, 1] -> [bs * nvars, patch_num, d_ff, 1]
        y = (hs @ C.unsqueeze(-1)).squeeze(3)

        y = y + D * x

        return y

# taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output