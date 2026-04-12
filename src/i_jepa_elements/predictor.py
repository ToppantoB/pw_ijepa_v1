from .transformer import TransformerEncoderBlock
import torch.nn as nn

class Predictor(nn.Module):
  def __init__(self, embed_dim=192, depth=3, mlp_dim=384, num_heads=3):
    super().__init__()
    self.transformer_blocks = nn.ModuleList([
      TransformerEncoderBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)
    ])
    self.norm = nn.LayerNorm(embed_dim)
    
  def forward(self, x):
    for block in self.transformer_blocks:
        x = block(x)
    x = self.norm(x)
    
    return x