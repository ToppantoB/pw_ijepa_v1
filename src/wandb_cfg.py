import wandb
from config import CONFIG

# Start a new wandb run to track this script.
wandb_logger = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="toppantob_org",
    # Set the wandb project where this run will be logged.
    project="pw-i_jepa-v1",
    name=f"Run v-{CONFIG.run_version}",
    # Track hyperparameters and run metadata.
    config={
        "architecture": "I-JEPA",
        "dataset": "STL-10",
        "image-and-patching": {
            "img_size": CONFIG.img_size,
            "patch_size": CONFIG.patch_size,
            "block_size": CONFIG.block_size,
            "number_of_blocks": CONFIG.number_of_blocks,
        },
        "encoder_config": {
            "encoder_embed_dim": CONFIG.encoder_embed_dim,
            "encoder_num_heads": CONFIG.encoder_num_heads,
            "encoder_depth": CONFIG.encoder_depth,
            "encoder_mlp_dim": CONFIG.encoder_mlp_dim,
            "min_context_size": CONFIG.min_context_size
        },
        "predictor_config": {
            "predictor_embed_dim": CONFIG.predictor_embed_dim,
            "predictor_depth": CONFIG.predictor_depth,
            "predictor_mlp_dim": CONFIG.predictor_mlp_dim,
            "predictor_num_heads": CONFIG.predictor_num_heads,
        },
        "training_config": {
            "epochs": CONFIG.epochs,
            "warmup_steps": CONFIG.warmup_steps,
            "batch_size": CONFIG.batch_size,
            "tau": CONFIG.tau_base,
            "tau_type": "fix",
            "encoder_learning_rate": CONFIG.base_learning_rate,
            "predictor_learning_rate_multiplier": CONFIG.predictor_lr_multiplier,
            "weight_decay_base": CONFIG.weight_decay_base,
            "weight_decay_max": CONFIG.weight_decay_max,
        },
    },
)
