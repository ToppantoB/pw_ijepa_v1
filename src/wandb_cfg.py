import wandb

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="toppantob_org",
    # Set the wandb project where this run will be logged.
    project="pw_vit-v1",
    # Track hyperparameters and run metadata.
    config={
        "architecture": "ViT",
        "dataset": "CIFAR-10",
        "epochs": 100,
    },
)
