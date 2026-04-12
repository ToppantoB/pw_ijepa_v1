import torch
import torch.nn as nn
import numpy as np

class PatchEmbedding(nn.Module):
  def __init__(self, img_size=96, patch_size=8, in_channels=3, embed_dim=768):
    super().__init__()
    self.patch_size = patch_size
    self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
  def forward(self, x):
    B, C, H, W = x.shape
    x = self.proj(x).flatten(2).transpose(1, 2)
    return x
  
    
class PositionalEncoding(nn.Module):
  def __init__(self, embed_dim, seq_len):
    super().__init__()
    # self.pos_embed = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim) * 0.02 )   # +1 for the cls token
    self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02 )
    
  def forward(self, x):
    return x + self.pos_embed
  
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    
    def forward(self, x):
        return self.attn(x, x, x)[0]
    
    
class TransformerEncoderBlock(nn.Module):
  def __init__(self, embed_dim, num_heads, mlp_dim):
    super().__init__()
    self.attn = MultiHeadAttention(embed_dim, num_heads)
    self.mlp = nn.Sequential(
      nn.Linear(embed_dim, mlp_dim),
      nn.GELU(),
      nn.Dropout(0),
      nn.Linear(mlp_dim, embed_dim),
      nn.Dropout(0),
    )
    self.norm1 = nn.LayerNorm(embed_dim)
    self.norm2 = nn.LayerNorm(embed_dim)
    self.dropout = nn.Dropout(0)

  def forward(self, x):
    x = x + self.dropout(self.attn(self.norm1(x)))
    x = x + self.mlp(self.norm2(x))
    return x
    

class VisionTransformerBase(nn.Module):
    def __init__(self, img_size=96, patch_size=8, embed_dim=192, num_heads=3, depth=6, mlp_dim=768):
      super().__init__()
      self.patch_embedding = PatchEmbedding(img_size, patch_size, 3, embed_dim)
      
      self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2, embed_dim) * 0.02 )
      
      # values = np.arange(0, 144).reshape(1, 144, 1)
      # self.pos_embed = torch.from_numpy(np.broadcast_to(values, (1, 144, 192))).to("cuda")
      
      # self.pos_encoding = PositionalEncoding(embed_dim, (img_size // patch_size) ** 2)
      self.transformer_blocks = nn.ModuleList([
        TransformerEncoderBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)
      ])
      # self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02 )
      # self.mlp_head = nn.Linear(embed_dim, num_classes)
      self.norm = nn.LayerNorm(embed_dim)
      self.pos_drop = nn.Dropout(0)
      
    def forward(self, x):
      raise NotImplementedError
    
    