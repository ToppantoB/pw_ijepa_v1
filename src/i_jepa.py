import torch
import torch.nn as nn

from i_jepa_elements.target_encoder import TargetEncoder
from i_jepa_elements.context_encoder import ContextEncoder
from i_jepa_elements.predictor import Predictor

from config import CONFIG


class I_JEPA(nn.Module):
    """Main I-JEPA model class that integrates the Context Encoder, Target Encoder, and Predictor components.
    It processes the input image through the context and target encoders, 
    and then uses the predictor to predict the target block embeddings based on the context embeddings and positional information.
    """
    def __init__(
        self,
        img_size=96,
        patch_size=8,
        encoder_embed_dim=192,
        encoder_num_heads=3,
        encoder_depth=6,
        encoder_mlp_dim=768,
        predictor_embed_dim=192,
        predictor_depth=3,
        predictor_mlp_dim=384,
        predictor_num_heads=3,
    ):
        super().__init__()

        self.context_encoder = ContextEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=encoder_embed_dim,
            num_heads=encoder_num_heads,
            depth=encoder_depth,
            mlp_dim=encoder_mlp_dim,
        )

        self.target_encoder = TargetEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=encoder_embed_dim,
            num_heads=encoder_num_heads,
            depth=encoder_depth,
            mlp_dim=encoder_mlp_dim,
            drop_path_rate=0.0,
        )

        # Freeze the target encoder parameters to prevent them from being updated during training
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.predictor = Predictor(
            embed_dim=predictor_embed_dim,
            depth=predictor_depth,
            mlp_dim=predictor_mlp_dim,
            num_heads=predictor_num_heads,
        )

        self.learnable_shared_vector = nn.Parameter(
            torch.zeros(1, 1, encoder_embed_dim)
        )
        
        # Initialize the learnable shared vector with a truncated normal distribution
        nn.init.trunc_normal_(self.learnable_shared_vector, std=0.02)

    def forward(self, input, target_blocks, context_crop, step):
        """Forward pass for the I-JEPA model.
        
        Args:
            input (torch.Tensor): The input image tensor of shape (batch_size, channels, height, width).
            target_blocks (torch.Tensor): An array of patch indices corresponding to the target blocks.
            context_crop (torch.Tensor): Crop information for the context encoder.
            step (int): The current training step.
        """
        with torch.no_grad():
            # calculate the target embedding
            target_embedding = self.target_encoder(input).detach()

        # metrics to monitor training process
        std_dev_list = []

        if step % 100 == 0:
            with torch.no_grad():
                # calculate standard deviation across batch and patches, then average over embed_dim for tracking collapse
                std_dev = target_embedding.std(dim=(0, 1)).mean().item()
                std_dev_list = [std_dev]

        # calculate the context embedding (image is masked within the context_encoder according to target_blocks)
        context_embedding = self.context_encoder(input, target_blocks, context_crop)

        # initialize loss
        total_loss = 0.0
        num_blocks = len(target_blocks)

        for current_target in target_blocks:
            # extract the embedding for the current target block
            current_ground_truth = target_embedding[:, current_target, :]

            # get the positional embedding for it...
            positional_embedding = self.context_encoder.pos_embed[:, current_target, :]
            # ...and enrich it with the shared vector
            enhanced_positional_embedding = (
                positional_embedding + self.learnable_shared_vector
            )

            # attach these positionally embededd patches to the output of the context encoder
            predicor_input = torch.concat(
                [context_embedding, enhanced_positional_embedding.expand(CONFIG.batch_size , -1, -1)],
                dim=1,
            )

            # get the prediction and...
            predictor_out = self.predictor(predicor_input)

            # ... extract the targets and ...
            target_size = current_target.shape[0]
            current_prediction = predictor_out[:, -target_size:, :]
            
            # ... calculate the loss
            block_loss = nn.functional.mse_loss(
                current_prediction, current_ground_truth, reduction="mean"
            )
                        
            total_loss += block_loss

        return total_loss / num_blocks, std_dev_list
