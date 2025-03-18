import torch
from torch import nn
from layers.Embed import PatchEmbedding, DataEmbedding_inverted,DataEmbedding,PatchEmbedding_inverted
from layers.CMambaEncoder import CMambaEncoder, CMambaBlock
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from mamba_ssm import Mamba
from layers.iCMamba_EncDec import Encoder, EncoderLayer

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_N, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_N)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [B, L, patch_num, d_model]
        x = self.flatten(x) # [B, L, patch_num*d_model]
        x = self.linear(x)  # [B, L, patch_num*d_model] -> [B, L, enc_in] = [B, L, N]
        x = self.dropout(x)
        return x

class to_pred_len(nn.Module):
    def __init__(self, L, pred_len, head_dropout=0):
        super().__init__()
        self.pred_len = pred_len
        self.linear = nn.Linear(L, pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [B, L, N]
        # print(f'x: {x.shape}')
        # print(f'pred_len: {self.pred_len}')
        x = x.transpose(1, 2)  # [B, N, L]
        x = self.linear(x)     # [B, N, L] -> [B, N, pred_len]
        x = self.dropout(x)
        x = x.transpose(1, 2)  # [B, pred_len, N]
        return x

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):

    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.padding = self.stride
        # patching and embedding
        configs.patch_num = int((configs.enc_in + self.padding - self.patch_len) / self.stride + 1)

        self.patch_embedding = PatchEmbedding_inverted(
            configs.d_model, self.patch_len, self.stride, self.padding, configs.head_dropout)

        # Encoder
        self.encoder = CMambaEncoder(configs)
        # Prediction Head
        self.head_nf = configs.d_model * configs.patch_num
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.enc_in,   # target_N为原来特征数
                                head_dropout=configs.head_dropout)
        #self.apply(init_weights)
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.linear = to_pred_len(configs.seq_len, configs.pred_len)
        self.decompsition = series_decomp(configs.moving_avg)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # print(f'x_enc: {x_enc.shape}x_mark_enc: {x_mark_enc.shape}x_dec: {x_dec.shape}x_mark_dec: {x_mark_dec.shape}')
        # Instance Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev  # [bs, l, N]
                        # x_mark: [bs, l, 4]

        seasonal_init, trend_init = self.decompsition(x_enc)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        # do patching and embedding
        # x_enc = x_enc.permute(0, 2, 1)  # [bs, L, N] -> [bs, N, L]
        # B N pn D
            # N: num of channels
            # pn: num of patches    pn= ⌊(N−P )/S ⌋ + 1, patch length P and stride S
            # D: d model
        enc_out, patch_num = self.patch_embedding(x_enc)    # [B*patch_num, L, d_model]
        
        enc_out = self.encoder(enc_out)     # [B*patch_num, L, d_model]
        enc_out = torch.reshape(enc_out, (-1, patch_num, enc_out.shape[-2], enc_out.shape[-1])) # 恢复形状: [B, patch_num, L, d_model]
        enc_out = enc_out.permute(0, 2, 1, 3)  # [B, L, patch_num, d_model]
        # 解码器 target_N变为N
        dec_out = self.head(enc_out)    # [B, L, target_N] 也就是  [B, L, N]
        dec_out = self.linear(dec_out)  # [B, pred_len, N]

        # De-Normalization
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        )
        # [B, pred_len, N]
        return dec_out