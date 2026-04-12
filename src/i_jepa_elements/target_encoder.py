from .transformer import VisionTransformerBase

class TargetEncoder(VisionTransformerBase):
  def forward(self, x):
      x = self.patch_embedding(x)
      
      x = x + self.pos_embed
            
      for block in self.transformer_blocks:
        x = block(x)
        
      x = self.norm(x)      
      
      return x

