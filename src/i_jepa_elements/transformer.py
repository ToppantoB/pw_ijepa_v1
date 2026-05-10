import torch
import torch.nn as nn
import numpy as np


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=8, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, seq_len):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)

    def forward(self, x):
        return x + self.pos_embed


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        return self.attn(x, x, x)[0]


# Drop path copied from original I-JEPA codebase -- start
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
# Drop path copied from original I-JEPA codebase -- end

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, drop_path_rate):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class VisionTransformerBase(nn.Module):
    def __init__(
        self,
        img_size=96,
        patch_size=8,
        embed_dim=192,
        num_heads=3,
        depth=6,
        mlp_dim=768,
        drop_path_rate=0.1,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(patch_size, 3, embed_dim)

        self.pos_embed = nn.Parameter(
            torch.randn(1, (img_size // patch_size) ** 2, embed_dim) * 0.02
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim, num_heads, mlp_dim, drop_path_rate=dpr[i]
                )
                for i in range(depth)
            ]
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_drop = nn.Dropout(0)

    def forward(self, x):
        raise NotImplementedError
