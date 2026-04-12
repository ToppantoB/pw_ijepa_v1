import random
import numpy as np
import torch
import math


def get_block(image_size=96, patch_size=8, block_size=4):
    """Returns the patch indices of a random "square-shaped" block from a "sqare-shaped" image

    Args:
      image_size (int, optional): The width of the image in pixels. Defaults to 96.
      patch_size (int, optional): Width of the patch in pixels. Defaults to 8.
      patch_size (int, optional): Width of the block. Defaults to 4.
    """
    if (patch_size * block_size) > image_size:
        raise Exception("patch_size * block_size cannot be larger than image_size")

    if image_size % patch_size != 0:
        raise Exception("image_size must be divisible by patch_size")

    image_size_in_patches = image_size // patch_size

    horizontal_start = random.randint(0, image_size_in_patches - block_size)
    vertical_start = random.randint(0, image_size_in_patches - block_size)

    # start_idx = v_start * image_size + h_start
    # first_row = np.arange(start_idx, start_idx + patch_size)

    # horizontal_start = random.randint(1, image_size - (patch_size - 1))
    # vertical_start = random.randint(1, image_size - (patch_size - 1))

    start_idx = vertical_start * image_size_in_patches + horizontal_start

    first_row = torch.arange(
        start_idx,
        start_idx + block_size,
        dtype=int
    )

    patch_matrix = first_row + torch.arange(block_size, dtype=int)[:, None] * image_size_in_patches

    return patch_matrix.flatten()


# TODO: check if zero-indexing is required in ViT
# TODO: maybe prevent having patches on the edge?


def get_blocks(image_size=96, patch_size=8, block_size=4, number_of_blocks=2):
    """Returns a list of patch indices of a random "square-shaped" block from a "sqare-shaped" image

    Args:
      image_size (int, optional): The width of the image in pixels. Defaults to 96.
      patch_size (int, optional): Width of the patch in pixels. Defaults to 8.
      block_size (int, optional): Width of the block. Defaults to 4.
      number_of_patches (int, optional): Number of patches to return. Defaults to 2.
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
    """
    img_size_after_crop = np.random.randint(min_img_size_after_crop, image_size + 1)
    
    amount_to_crop = image_size - img_size_after_crop
     
    rows_to_crop = []
    cols_to_crop = []
    
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
        
    for row in rows_to_crop:
        patches_to_crop.append(first_row_idx + (image_size * row))
    
    for col in cols_to_crop:
        patches_to_crop.append(first_col_idx + col)
    
    if len(patches_to_crop):    
        return torch.unique(torch.cat(patches_to_crop, dim=0))
    else:
        return []


def update_target_encoder(context_encoder, target_encoder, tau):
    with torch.no_grad():
        for context_params, target_params in zip(context_encoder.parameters(), target_encoder.parameters()):
            target_params.data.mul_(tau).add_(context_params.data, alpha=1.0 - tau)
            
            
def get_current_tau(step, total_steps, base_tau=0.996, max_tau=1.0):
    return max_tau - (max_tau - base_tau) * (1 + math.cos(math.pi * step / total_steps)) / 2