from .transformer import TransformerEncoderBlock
import torch.nn as nn
import torch

class Predictor(nn.Module):
    """Predictor class for the I-JEPA model. It consists of a series of transformer encoder blocks followed by a layer normalization."""
    def __init__(
        self, embed_dim=192, depth=3, mlp_dim=384, num_heads=3, drop_path_rate=0.1
    ):
        """Initializes the Predictor model with the specified parameters."""
        super().__init__()

        _drop_path_rate = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim, num_heads, mlp_dim, drop_path_rate=_drop_path_rate[i]
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """Forward pass for the Predictor.
        
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, num_patches, embed_dim).
            
        Returns:
            torch.Tensor: The output features from the Predictor after processing through the transformer blocks.
        """
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)

        return x