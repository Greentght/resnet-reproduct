import torch
from torch import nn
from torch.nn import functional as F
import math
from self_attention import ScaledDotProductAttention, MultiHeadAttention 

class AddNorm(nn.Module):
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(norm_shape)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, Y):
        return self.ln(X + self.dropout(Y))

class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(F.relu(self.dense1(X)))


class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False):
        super().__init__()
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 num_layers, dropout, use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.blocks = nn.Sequential()
        for i in range(num_layers):
            self.blocks.add_module(f"block{i}",
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, valid_lens):
        original_ndim = X.dim()
        if original_ndim == 2:
            X = X.unsqueeze(0) 

            if valid_lens is None:
                valid_lens = torch.tensor([X.shape[1]], device=X.device)
            elif valid_lens.dim() == 0:
                valid_lens = valid_lens.unsqueeze(0)
                 
        elif original_ndim != 3:
            raise ValueError(f"输入张量必须是 2D 或 3D，但收到 {original_ndim}D 张量。")

        for blk in self.blocks:
            X = blk(X, valid_lens)
        if original_ndim == 2:
            X = X.squeeze(0)

        return X

d_model = 4      
num_heads = 4         
num_layers = 2        
dropout_rate = 0.1
seq_len = 10          
feature_dim = d_model 

key_size = d_model
query_size = d_model
value_size = d_model
ffn_num_input = d_model
ffn_num_hiddens = d_model * 4
norm_shape = [d_model] 


encoder = TransformerEncoder(
    key_size, query_size, value_size, d_model,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout_rate)
encoder.eval() 


X_raw = torch.randn(seq_len, feature_dim) 
valid_lens = None 

print(f"输入张量 X {X_raw}")

output = encoder(X_raw, valid_lens)

print(f"编码器输出{output}")
