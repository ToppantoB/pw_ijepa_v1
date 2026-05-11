import random
import numpy as np
import torch
import math
from config import CONFIG
from pathlib import Path
import warnings


def get_block(image_size=96, patch_size=8, block_size=4):
    """Returns the patch indices of a random "square-shaped" block from a "sqare-shaped" image

    Args:
      image_size (int, optional): The width of the image in pixels. Defaults to 96.
      patch_size (int, optional): Width of the patch in pixels. Defaults to 8.
      patch_size (int, optional): Width of the block. Defaults to 4.
      
    Returns:
      torch.Tensor: A tensor containing the patch indices corresponding to the randomly selected block.
    """
    if (patch_size * block_size) > image_size:
        raise Exception("patch_size * block_size cannot be larger than image_size")

    if image_size % patch_size != 0:
        raise Exception("image_size must be divisible by patch_size")

    image_size_in_patches = image_size // patch_size

    horizontal_start = random.randint(0, image_size_in_patches - block_size)
    vertical_start = random.randint(0, image_size_in_patches - block_size)

    start_idx = vertical_start * image_size_in_patches + horizontal_start

    first_row = torch.arange(
        start_idx,
        start_idx + block_size,
        dtype=int
    )

    patch_matrix = first_row + torch.arange(block_size, dtype=int)[:, None] * image_size_in_patches

    return patch_matrix.flatten()

def get_blocks(image_size=96, patch_size=8, block_size=4, number_of_blocks=2):
    """Returns a list of patch indices of a random "square-shaped" block from a "sqare-shaped" image

    Args:
      image_size (int, optional): The width of the image in pixels. Defaults to 96.
      patch_size (int, optional): Width of the patch in pixels. Defaults to 8.
      block_size (int, optional): Width of the block. Defaults to 4.
      number_of_blocks (int, optional): Number of blocks to return. Defaults to 2.
      
    Returns:
        torch.Tensor: A tensor containing the patch indices corresponding to the randomly selected blocks.
    """
    target_blocks = [
        get_block(image_size, patch_size, block_size) for _ in range(number_of_blocks)
    ]

    return torch.stack(target_blocks)

def get_image_crop(image_size=12, min_img_size_after_crop=10):
    """Returns a list of patch indices of a "square-shaped" crop from a "square-shaped" image
    
    Args:
      image_size (int, optional): The width of the image in patches. Defaults to 12.
      img_size_after_crop (int, optional): The width of the image in patches after the crop. Defaults to 10.
      
    Returns:
      torch.Tensor: A tensor containing the patch indices corresponding to the cropped area of the image.
    """
    img_size_after_crop = np.random.randint(min_img_size_after_crop, image_size + 1)
    
    amount_to_crop = image_size - img_size_after_crop
     
    rows_to_crop = []
    cols_to_crop = []
    
    # initialize the vertical and horizontal crop boundaries to the full image size
    v_start, v_end = 0, image_size -1
    h_start, h_end = 0, image_size -1
    
    for _ in range(amount_to_crop):
        # decide whether to crop from beginning or end
        vertical_crop = random.randint(0, 1)
        if vertical_crop:
            rows_to_crop.append(v_start)
            v_start += 1
        else:
            rows_to_crop.append(v_end)
            v_end -= 1
        
        
        # decide whether to crop from beginning or end
        horizontal_crop = random.randint(0, 1)
        if horizontal_crop:
            cols_to_crop.append(h_start)
            h_start += 1
        else:
            cols_to_crop.append(h_end)
            h_end -= 1        

    first_row_idx = torch.arange(0, image_size)
    first_col_idx = first_row_idx * image_size
    
    patches_to_crop = []
    
    # calculate the patch indices based on the selected rows and columns to crop
    for row in rows_to_crop:
        patches_to_crop.append(first_row_idx + (image_size * row))
    
    for col in cols_to_crop:
        patches_to_crop.append(first_col_idx + col)
    
    if len(patches_to_crop):    
        return torch.unique(torch.cat(patches_to_crop, dim=0))
    else:
        return []


def update_target_encoder(context_encoder, target_encoder, tau):
    """Updates the target encoder parameters using an exponential moving average (EMA).

    Args:
      context_encoder (torch.nn.Module): The actively trained context encoder.
      target_encoder (torch.nn.Module): The target encoder to be updated in-place.
      tau (float): The momentum parameter controlling the EMA update rate.
    """
    with torch.no_grad():
        for context_params, target_params in zip(context_encoder.parameters(), target_encoder.parameters()):
            target_params.data.mul_(tau).add_(context_params.data, alpha=1.0 - tau)
            
            
def get_current_tau(step, total_steps, base_tau=0.996, max_tau=1.0):
    """Calculates the current momentum parameter (tau) using a cosine schedule.

    Args:
      step (int): The current training step.
      total_steps (int): The total number of training steps.
      base_tau (float, optional): The initial starting value for tau. Defaults to 0.996.
      max_tau (float, optional): The maximum and final value for tau. Defaults to 1.0.

    Returns:
      float: The scheduled tau value for the current training step.
    """
    return max_tau - (max_tau - base_tau) * (1 + math.cos(math.pi * step / total_steps)) / 2

def get_parameter_groups(model):
    """Separates model parameters into optimizer groups, selectively applying weight decay.

    Args:
      model (torch.nn.Module): The main model containing the context encoder and predictor.

    Returns:
      list: A list of dictionaries defining the parameter groups, learning rates, and weight decay schedules for the optimizer.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.context_encoder.named_parameters():
        if not param.requires_grad:
            continue
        
        # exclude parameters that are 1-dimensional (like LayerNorm weights) or biases from weight decay
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {
            "params": decay_params,
            "lr": CONFIG.base_learning_rate,
            "weight_decay": 0.01,
            "apply_wd_schedule": True 
        },
        {
            "params": no_decay_params,
            "lr": CONFIG.base_learning_rate,
            "weight_decay": 0.0,
            "apply_wd_schedule": False
        },
        {
            "params": model.predictor.parameters(),
            "lr": CONFIG.base_learning_rate * CONFIG.predictor_lr_multiplier,
            "weight_decay": 0.0,
            "apply_wd_schedule": False
        },
    ]
    
def get_project_root(anchor_file="requirements.txt"):
    """
    Traverses upwards from the current file to find the project root 
    based on the presence of a specific anchor file/directory.
    
    Args:
      anchor_file (str, optional): The name of the file or directory that indicates the project root. Defaults to "requirements.txt".
      
    Returns:
      Path: The path to the project root.
    """
    current_path = Path(__file__).resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / anchor_file).exists():
            return parent
        
    return current_path.parent

def get_compute_device():
    """
    Returns the available computational device. Shows a warning in case CUDA is not available.
    
    Returns:
      str: The device to perform computations on, either "cuda" or "cpu".
    """
    if torch.cuda.is_available():
        return "cuda"
    else:
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        
        warnings.warn(
            "\n" + "="*60 + 
            f"{YELLOW}\nWARNING: CUDA is not available. Running on CPU is possible but highly impractical "
            f"for this workload. It is strongly recommended to use an GPU with CUDA.{RESET}"
            "\n" + "="*60,
            UserWarning
        )
        return "cpu"