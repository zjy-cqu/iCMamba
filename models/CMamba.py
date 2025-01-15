import torch
from torch import nn
from layers.Embed import PatchEmbedding, DataEmbedding_inverted,DataEmbedding
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
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs, nvars, d_model, patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    
class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride
        # patching and embedding
        configs.patch_num = int((configs.seq_len - patch_len) / stride + 2)
        # self.patch_embedding = PatchEmbedding(
            # configs.d_model, patch_len, stride, padding, configs.head_dropout)
        # self.enc_embedding = DataEmbedding_inverted(
        # #     configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        # Encoder
        self.cencoder = CMambaEncoder(configs)

        self.output_attention = configs.output_attention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=0.1,
                        output_attention=self.output_attention),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    Mamba(
                        d_model=configs.d_model,  # Model dimension d_model
                        d_state=configs.d_state,  # SSM state expansion factor
                        d_conv=4,  # Local convolution width
                        expand=2,  # Block expansion factor)
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Prediction Head
        self.head_nf = configs.d_model * configs.patch_num
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.head_dropout)
        #self.apply(init_weights)
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Instance Normalization
        # B V L
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev  # [bs, l, N]
                        # x_mark: [bs, l, 4]
        _, _, D = x_enc.shape

        enc_out = self.enc_embedding(x_enc, x_mark_enc) 
        # enc_out = self.cencoder(x_enc, x_mark_enc) # [bs, 11, 128]
        # print("before ENCODER: enc_out.shape", enc_out.shape)
        enc_out, attn = self.encoder(enc_out)                 # [64, 11, 128]
        # print("after ENCODER: enc_out.shape", enc_out.shape)


        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :D]    # # [64, 96, 7]  B L D

        return dec_out