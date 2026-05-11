from .transformer import VisionTransformerBase

class TargetEncoder(VisionTransformerBase):
  """Target Encoder class for the I-JEPA model. It is based on the VisionTransformerBase class."""
  def forward(self, x):
      """Forward pass for the Target Encoder.
      Args:
          x (torch.Tensor): The input image tensor of shape (batch_size, channels, height, width).  
      
      Returns:
          torch.Tensor: The output features from the Target Encoder after processing through the transformer blocks.
      """
      x = self.patch_embedding(x)
      
      x = x + self.pos_embed
            
      for block in self.transformer_blocks:
        x = block(x)
        
      x = self.norm(x)      
      
      return x

