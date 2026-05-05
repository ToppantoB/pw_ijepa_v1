from i_jepa import I_JEPA
from config import CONFIG
from utils import (
    get_blocks,
    update_target_encoder,
    get_image_crop,
    get_current_tau,
    compute_linear_weight_decay,
    get_parameter_groups
)
from eval_scripts.eval import do_eval

import wandb
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch.optim.lr_scheduler"
)

import pynvml
import sys
import torch.optim as optim
import os
import numpy as np
import torch

from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm
from dataset import get_train_loader

from wandb_cfg import wandb_logger

pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

run_version = f"v{CONFIG.run_version}"
knn_acc = 0

OUT_DIR_BEST = os.getenv("OUT_DIR", f"./outputs/{run_version}/best")
OUT_DIR_ALL = os.getenv("OUT_DIR", f"./outputs/{run_version}/all")

os.makedirs(OUT_DIR_BEST, exist_ok=True)
os.makedirs(OUT_DIR_ALL, exist_ok=True)


def main():
    train_loader = get_train_loader()

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
    model.to("cuda")

    total_steps = CONFIG.epochs * (len(train_loader) / CONFIG.grad_accum_steps)

    # learning_rate = CONFIG.base_learning_rate * (CONFIG.batch_size / 32)

    # optimizer = optim.AdamW(
    #     [
    #         {
    #             "params": model.context_encoder.parameters(),
    #             "lr": CONFIG.base_learning_rate,
    #         },
    #         {
    #             "params": model.predictor.parameters(),
    #             "lr": CONFIG.base_learning_rate * CONFIG.predictor_lr_multiplier,
    #         },
    #     ],
    #     weight_decay=0.01,
    # )

    optimizer = optim.AdamW(get_parameter_groups(model))

    steps_per_epoch = len(train_loader) / CONFIG.grad_accum_steps
    actual_warmup_steps = int(CONFIG.warmup_steps * steps_per_epoch)

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=actual_warmup_steps
    )
    decay_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=((CONFIG.epochs * steps_per_epoch) - actual_warmup_steps),
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[actual_warmup_steps],
    )

    best_loss = np.inf
    accum_lost = 0
    std_devs_accum_list = []
    features = []

    for epoch in range(CONFIG.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG.epochs}")
        optimizer.zero_grad()

        for step, (inputs, _) in enumerate(pbar):
            inputs = inputs.to("cuda")

            context_crop = get_image_crop(
                image_size=int(CONFIG.img_size / CONFIG.patch_size),
                min_img_size_after_crop=CONFIG.min_context_size,
            )

            target_blocks = get_blocks(
                image_size=CONFIG.img_size,
                patch_size=CONFIG.patch_size,
                block_size=CONFIG.block_size,
                number_of_blocks=CONFIG.number_of_blocks,
            )

            loss, metrics = model(inputs, target_blocks, context_crop, step)
            accum_lost += loss.item()
            std_devs_accum_list += metrics[0]

            loss.backward()

            if (step + 1) % CONFIG.grad_accum_steps == 0 or (step + 1) == len(
                train_loader
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step = (epoch * len(train_loader) / CONFIG.grad_accum_steps) + (
                    step // CONFIG.grad_accum_steps
                )
                current_tau = get_current_tau(
                    global_step, total_steps, CONFIG.tau_base, CONFIG.tau_end
                )

                update_target_encoder(
                    model.context_encoder, model.target_encoder, tau=current_tau
                )

                #################
                # current_wd = compute_linear_weight_decay(
                #     global_step,
                #     total_steps,
                #     CONFIG.weight_decay_base,
                #     CONFIG.weight_decay_max,
                # )
                
                # for param_group in optimizer.param_groups:
                #     if param_group.get("apply_wd_schedule", False):
                #         param_group["weight_decay"] = current_wd
                ##################


            if step % 100 == 0:
                temp = pynvml.nvmlDeviceGetTemperature(
                    gpu_handle, pynvml.NVML_TEMPERATURE_GPU
                )
                if temp > 87:
                    print(
                        f"CRITICAL: GPU temperature reached {temp}°C. Stopping training."
                    )

                    pynvml.nvmlShutdown()
                    sys.exit(1)

                features = metrics[1].to("cpu")

        avg_loss = accum_lost / len(train_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            param_version = epoch
            save_type = "best"
            path_to_save = f"{OUT_DIR_BEST}/ijepa_stl10_{epoch}.pt"
            torch.save(model.state_dict(), path_to_save)
            print(f"Saved: {path_to_save} (acc={best_loss:.4f})")
        elif epoch % 10 == 0:
            param_version = epoch
            save_type = "all"
            path_to_save = f"{OUT_DIR_ALL}/ijepa_stl10_{epoch}.pt"
            torch.save(model.state_dict(), path_to_save)

        if epoch % 10 == 0:
            knn_acc = do_eval(
                CONFIG.run_version, param_version, eval_type="knn", save_type=save_type
            )

        if epoch % 20 == 0 or (epoch + 1) == CONFIG.epochs:
            lin_probe_acc = do_eval(
                CONFIG.run_version,
                param_version,
                eval_type="linear_probe",
                save_type=save_type,
            )

        wandb_logger.log(   
            {
                "Loss": avg_loss,
                "Standard deviation of img tensors": sum(std_devs_accum_list)
                / len(std_devs_accum_list),
                "Feature distribution": wandb.Histogram(features),
                "k-NN Accuracy": knn_acc,
                "Linear probe Accuracy": lin_probe_acc,
            },
            step=epoch,
        )

        std_devs_accum_list = []
        accum_lost = 0

    pynvml.nvmlShutdown()
    wandb_logger.finish()


if __name__ == "__main__":
    main()
