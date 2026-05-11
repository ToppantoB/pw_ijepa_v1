from i_jepa_elements.context_encoder import ContextEncoder
from config import CONFIG
import torch


def get_encoder(path_to_params, encoder_config, device):
    """Initializes a ContextEncoder model and loads its pre-trained weights from a file.

    Args:
      path_to_params (str): The file path to the saved model parameters (state dict).
      encoder_config (dict): Configuration dictionary containing 'patch_size' and 'depth' for the encoder.
      device (str or torch.device): The device (e.g., 'cuda' or 'cpu') to perform computations on.

    Returns:
      ContextEncoder: The initialized encoder model, loaded with the specified weights and moved to the GPU.
    """
    context_encoder = ContextEncoder(
        img_size=CONFIG.img_size,
        patch_size=encoder_config["patch_size"],
        embed_dim=CONFIG.encoder_embed_dim,
        num_heads=CONFIG.encoder_num_heads,
        depth=encoder_config["depth"],
        mlp_dim=CONFIG.encoder_mlp_dim,
    )

    state_dict = torch.load(path_to_params, weights_only=True)

    prefix = "context_encoder."
    vit_state_dict = {
        k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)
    }

    context_encoder.to(device)
    context_encoder.load_state_dict(state_dict=vit_state_dict)

    return context_encoder
