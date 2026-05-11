from .transformer import VisionTransformerBase
import torch
import numpy as np


class ContextEncoder(VisionTransformerBase):
    """Context Encoder class for the I-JEPA model. It is based on the VisionTransformerBase class."""
    
    def forward(self, x, target_blocks=np.array([]), context_crop=np.array([])):
        """Forward pass for the Context Encoder.
        
        Args:
            x (torch.Tensor): The input image tensor of shape (batch_size, channels, height, width).
            target_blocks (np.array, optional): An array of patch indices corresponding to the target blocks. Defaults to an empty array.
            context_crop (np.array, optional): An array of patch indices corresponding to the context crop. Defaults to an empty array.
            
        Returns:
            torch.Tensor: The output features from the Context Encoder after processing through the transformer blocks.
        """
        x = self.patch_embedding(x)

        x = x + self.pos_embed

        # Build the mask to remove the target blocks and context crop from the input
        mask = torch.ones(x.shape[1], dtype=torch.bool)
        mask[np.unique(target_blocks)] = False
        mask[context_crop] = False

        # Apply the mask
        x = x[:, mask, :]

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)

        return x
