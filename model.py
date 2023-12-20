import torch
import torch.nn as nn
from termcolor import colored

config = {
'vocab_size' : 100277,  # adjust based on your dataset
'd_model' : 512,
'num_heads' : 8,
'num_layers' : 6,
'd_ff' : 1025,
'max_len' : 128}

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, d_model),requires_grad=False)
        self.positional_encoding[:, :, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[:, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        return x + self.positional_encoding[:, :x.size(1)].detach()

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # Linear transformation for query, key, and value
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        # print(colored(f"score shape {scores.shape}","cyan"))
        if mask is not None:
            scores = torch.tril(scores)
            scores = scores.masked_fill(scores == 0, float('-inf'))

        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # Concatenate and linear transformation
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(attention_output)

        return output

class Feedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(Feedforward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.feedforward = Feedforward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Multi-Head Self Attention
        attention_output = self.self_attention(x, x, x, mask)
        x = x + self.dropout1(attention_output)
        x = self.norm1(x)

        # Feedforward
        ff_output = self.feedforward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([TransformerDecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x
 
class TransformerDecoderModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout=0.1):
        super(TransformerDecoderModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.decoder = TransformerDecoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)


    def forward(self, x):

        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.decoder(x,mask=True)
        x = self.fc_out(x)
        return x


