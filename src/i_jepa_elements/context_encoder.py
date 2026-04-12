from .transformer import VisionTransformerBase
import torch
import numpy as np


class ContextEncoder(VisionTransformerBase):
    def forward(self, x, target_blocks=np.array([]), context_crop=np.array([])):
        x = self.patch_embedding(x)

        x = x + self.pos_embed

        mask = torch.ones(x.shape[1], dtype=torch.bool)
        mask[np.unique(target_blocks)] = False
        mask[context_crop] = False

        x = x[:, mask, :]

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)

        return x
