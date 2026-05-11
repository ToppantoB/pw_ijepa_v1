from i_jepa import I_JEPA
from config import CONFIG
from utils import (
    get_blocks,
    update_target_encoder,
    get_image_crop,
    get_current_tau,
    get_parameter_groups,
    get_project_root
)
from eval_scripts.eval import do_eval

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch.optim.lr_scheduler"
)

# import pynvml
import sys
import torch.optim as optim
import os
import numpy as np
import torch

from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm

from dataset import get_train_loader
from wandb_cfg import get_wandb_logger

# pynvml.nvmlInit()
# gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set up output directories for saving model checkpoints
project_root = get_project_root()

run_version = f"v{CONFIG.run_version}"
best_path = os.path.join(project_root, "outputs", run_version, "best")
latest_path = os.path.join(project_root, "outputs", run_version, "latest")

OUT_DIR_BEST = os.getenv("OUT_DIR", best_path)
OUT_DIR_LATEST = os.getenv("OUT_DIR", latest_path)

os.makedirs(OUT_DIR_BEST, exist_ok=True)
os.makedirs(OUT_DIR_LATEST, exist_ok=True)


def train(device):
    """
    Trains the I-JEPA model. All of the training configuration can be found in the CONFIG object
    
    device (str or torch.device): The device (e.g., 'cuda' or 'cpu') to perform computations on.
    """
    # load the training data
    train_loader = get_train_loader()
    # initialize weights & biases logger
    wandb_logger = get_wandb_logger()

    # initialize the model
    model = I_JEPA(
        img_size=CONFIG.img_size,
        patch_size=CONFIG.patch_size,
        encoder_embed_dim=CONFIG.encoder_embed_dim,
        encoder_num_heads=CONFIG.encoder_num_heads,
        encoder_depth=CONFIG.encoder_depth,
        encoder_mlp_dim=CONFIG.encoder_mlp_dim,
        predictor_embed_dim=CONFIG.predictor_embed_dim,
        predictor_depth=CONFIG.predictor_depth,
        predictor_mlp_dim=CONFIG.predictor_mlp_dim,
        predictor_num_heads=CONFIG.predictor_num_heads,
    )
    model.to(device)

    steps_per_epoch = len(train_loader) / CONFIG.grad_accum_steps
    
    # calculate total training steps for tau scheduling
    total_steps = CONFIG.epochs * steps_per_epoch

    optimizer = optim.AdamW(get_parameter_groups(model))

    # adjust warmup steps to be in terms of actual optimizer steps (accounting for grad accumulation)
    actual_warmup_steps = int(CONFIG.warmup_steps * steps_per_epoch)

    # set up learning rate schedulers: linear warmup followed by cosine decay
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=actual_warmup_steps
    )
    decay_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(total_steps - actual_warmup_steps),
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[actual_warmup_steps],
    )

    # variables to track best loss and accuracy for saving checkpoints
    best_loss = np.inf
    accum_lost = 0
    knn_acc = 0
    std_devs_accum_list = []

    for epoch in range(CONFIG.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG.epochs}")
        optimizer.zero_grad()

        for step, (inputs, _) in enumerate(pbar):
            inputs = inputs.to(device)

            # generate random crop for context encoder (ensuring minimum size after cropping)
            context_crop = get_image_crop(
                image_size=int(CONFIG.img_size / CONFIG.patch_size),
                min_img_size_after_crop=CONFIG.min_context_size,
            )

            # generate random target blocks for the target encoder
            target_blocks = get_blocks(
                image_size=CONFIG.img_size,
                patch_size=CONFIG.patch_size,
                block_size=CONFIG.block_size,
                number_of_blocks=CONFIG.number_of_blocks,
            )

            # forward pass through the model to compute loss and standard deviation metrics
            loss, std_dev_list = model(inputs, target_blocks, context_crop, step)
            accum_lost += loss.item()
            std_devs_accum_list += std_dev_list

            loss.backward()

            # adjust optimizer step and learning rate scheduler according to gradient accumulation
            if (step + 1) % CONFIG.grad_accum_steps == 0 or (step + 1) == len(
                train_loader
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step = (epoch * steps_per_epoch) + (
                    step // CONFIG.grad_accum_steps
                )
                
                # calculate the current tau value using a cosine schedule based on the global training step
                current_tau = get_current_tau(
                    global_step, total_steps, CONFIG.tau_base, CONFIG.tau_end
                )

                # update the target encoder parameters using an exponential moving average
                update_target_encoder(
                    model.context_encoder, model.target_encoder, tau=current_tau
                )

            if step % 100 == 0:
                # GPU monitoring for local training...
                # temp = pynvml.nvmlDeviceGetTemperature(
                #     gpu_handle, pynvml.NVML_TEMPERATURE_GPU
                # )
                # if temp > 87:
                #     print(
                #         f"CRITICAL: GPU temperature reached {temp}°C. Stopping training."
                #     )

                #     pynvml.nvmlShutdown()
                #     sys.exit(1)
                # ...GPU monitoring end
                pass

        avg_loss = accum_lost / len(train_loader)

        # save model checkpoints based on best loss and at regular intervals, and perform evaluations
        if avg_loss < best_loss:
            best_loss = avg_loss
            param_version = epoch
            save_type = "best"
            path_to_save = f"{OUT_DIR_BEST}/ijepa_stl10_{epoch}.pt"
            torch.save(model.state_dict(), path_to_save)
            print(f"Saved: {path_to_save} (acc={best_loss:.4f})")
        elif epoch % 10 == 0:
            param_version = epoch
            save_type = "latest"
            path_to_save = f"{OUT_DIR_LATEST}/ijepa_stl10_{epoch}.pt"
            torch.save(model.state_dict(), path_to_save)

        # perform k-NN evaluation every 10 epochs
        if epoch % 10 == 0:
            knn_acc = do_eval(
                CONFIG.run_version, param_version, eval_type="knn", save_type=save_type, device=device
            )

        # perform linear probe evaluation every 20 epochs and at the end of training
        if epoch % 20 == 0 or (epoch + 1) == CONFIG.epochs:
            lin_probe_acc = do_eval(
                CONFIG.run_version,
                param_version,
                eval_type="linear_probe",
                save_type=save_type,
                device=device
            )

        # log training metrics to Weights & Biases
        wandb_logger.log(   
            {
                "Loss": avg_loss,
                "Standard deviation of img tensors": sum(std_devs_accum_list)
                / len(std_devs_accum_list),
                "k-NN Accuracy": knn_acc,
                "Linear probe Accuracy": lin_probe_acc,
            },
            step=epoch,
        )

        std_devs_accum_list = []
        accum_lost = 0

    # pynvml.nvmlShutdown()
    wandb_logger.finish()