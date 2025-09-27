import torch
from torch import nn
from torch.nn import functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, mask=None):
        d_k = K.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_hiddens = num_hiddens
        
        assert num_hiddens % num_heads == 0
        
        self.d_k = num_hiddens // num_heads
        self.d_v = num_hiddens // num_heads

        self.W_q = nn.Linear(query_size, num_hiddens, bias=use_bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=use_bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=use_bias)
        
        self.attention = ScaledDotProductAttention(dropout)
        
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=use_bias)

    def transpose_qkv(self, X):
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        X = X.transpose(1, 2)
        return X

    def transpose_output(self, X):
        X = X.transpose(1, 2)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def forward(self, Q, K, V, valid_lens=None, mask=None):
        
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        Q = self.transpose_qkv(Q)
        K = self.transpose_qkv(K)
        V = self.transpose_qkv(V)
        
        if valid_lens is not None:
            valid_lens = valid_lens.unsqueeze(1).unsqueeze(2)
            padding_mask = torch.arange(K.shape[-2], device=K.device)[None, None, None, :] < valid_lens
        else:
            padding_mask = None

        final_mask = padding_mask if padding_mask is not None else mask
        if mask is not None and padding_mask is not None:
            final_mask = padding_mask & mask

        output = self.attention(Q, K, V, mask=final_mask) 
        
        output_concat = self.transpose_output(output) 

        return self.W_o(output_concat)

