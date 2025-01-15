import concurrent.futures
import threading

import torch.nn as nn
import torch.nn.functional as F
import torch
from mamba_ssm import Mamba
from layers.SelfAttention_Family import FullAttention, AttentionLayer

class EncoderLayer(nn.Module):
    def __init__(self, attention, mamba, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        self.attention = attention  # 自注意力用于学习变量间关系
        self.mamba = mamba         # Mamba用于学习时间依赖关系
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # 使用自注意力模块学习变量间的关系
        # print("encoder layer: x.shape", x.shape)
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + new_x
        attn_out = self.norm1(x)
        # attn_out = self.attention(x, attn_mask=attn_mask)
        # print("=============", x.shape)
        # 使用Mamba模块学习时间依赖关系
        time_out = self.norm2(self.mamba(x))

        # 将自注意力和Mamba输出结合
        # x = x + attn_out + time_out
        x = x + attn_out + time_out
        # y = x = self.norm2(x)

        # 不用FNN
        # y = self.dropout(self.activation(x.transpose(-1, 1)))
        # y = self.dropout(y.transpose(-1, 1))
        # 用FNN
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return x + y, attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

