import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from .operation import Conv1D, mask_logits


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images. (To 1D sequences)
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        """
        Args:
            x: torch.tensor, (batch_size, L, d)
            mask: torch.tensor, (batch_size, L), with 1 as valid

        Returns:

        """
        assert mask is not None
        x_embed = mask.cumsum(1, dtype=torch.float32)  # (bsz, L)
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)
        pos_x = x_embed[:, :, None] / dim_t  # (bsz, L, num_pos_feats)
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)  # (bsz, L, num_pos_feats*2)
        # import ipdb; ipdb.set_trace()
        return pos_x  # .permute(0, 2, 1)  # (bsz, num_pos_feats*2, L)
    
class TransformerPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, dim, 2).float() *
                    -(math.log(10000.0) / dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class PositionalEmbedding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, embedding_dim, num_embeddings):
        super(PositionalEmbedding, self).__init__()
        self.position_embeddings = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, inputs):
        bsz, seq_length = inputs.shape[:2]
        position_ids = torch.arange(seq_length,
                                    dtype=torch.long,
                                    device=inputs.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        return position_embeddings


class Projection(nn.Module):
    def __init__(self, in_dim, dim, drop_rate=0.0):
        super(Projection, self).__init__()
        self.drop = nn.Dropout(p=drop_rate)
        self.projection = Conv1D(in_dim=in_dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 bias=True,
                                 padding=0)
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, input_features):
        # the input feature with shape (batch_size, seq_len, in_dim)
        input_features = self.drop(input_features)
        output = self.projection(input_features)  # (batch_size, seq_len, dim)
        output = self.layer_norm(output)
        return output


class Prediction(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, drop_rate=0.):
        super(Prediction, self).__init__()
        self.fc1 = Conv1D(in_dim=in_dim,
                          out_dim=hidden_dim,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.fc2 = Conv1D(in_dim=hidden_dim,
                          out_dim=out_dim,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True)

    def forward(self, input_feature):
        output = self.fc1(input_feature)
        output = F.gelu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output

class MLP(nn.Module):

    def __init__(self, dims, dropout=0.1) -> None:
        super().__init__()
        # assert num_layers > 1, "this class is intended for multiple linear layers"
        # dims = dims
        num_layers = len(dims) - 1
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)])
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx != len(self.layers) - 1:
                x = F.gelu(x)
                x = self.do(x)
        return x