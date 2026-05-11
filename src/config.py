from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Image & Patching
    img_size: int = 96
    """The size of the input square image (e.g., 96 for a 96x96 image)."""

    patch_size: int = 8
    """The side length of the square patches the image is divided into."""

    block_size: int = 4
    """The side length of the square blocks in patches."""

    number_of_blocks: int = 5
    """The number of target blocks fot I-JEPA's training"""
        
    min_context_size: int = 10
    """Width of the context image in patches after initial crop"""
    
    # Training Settings
    epochs: int = 220
    """Number of epochs to train for"""
    
    batch_size: int = 128
    """The training batch size"""
    
    warmup_steps: int = 20
    """Warmup steps for training"""
    
    tau_base: float = 0.985
    """Initial value of Tau for the EMA"""
    
    tau_end: float = 1.0
    """End value of Tau for the EMA"""
    
    base_learning_rate: float = 1.5e-4
    """Base learning rate for Context Encoder"""
    
    grad_accum_steps: int = 2
    """Number of steps to do graident accumulation for"""
    
    predictor_lr_multiplier = 2
    """Factor to increase the learning rate predictor by"""
    
    # Encoder Settings
    encoder_embed_dim: int = 192
    """The dimension of the latent space for the encoder."""

    encoder_num_heads: int = 3
    """Number of self-attention heads in each encoder block."""

    encoder_depth: int = 12
    """The number of transformer layers in the encoder."""

    encoder_mlp_dim: int = 768
    """The hidden dimension size of the feed-forward network in the encoder."""

    # Predictor Settings
    predictor_embed_dim: int = 192
    """The dimension of the latent space for the predictor."""

    predictor_depth: int = 2
    """The number of transformer layers in the predictor."""

    predictor_mlp_dim: int = 768
    """The hidden dimension size of the feed-forward network in the predictor."""

    predictor_num_heads: int = 3
    """Number of self-attention heads in each predictor block."""
    
    # Additional
    run_version: int = 1
    """Version of the current run for WandB logs and artifact naming"""

CONFIG = Config()
