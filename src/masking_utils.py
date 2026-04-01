import random
import numpy as np

def get_patch(image_size=96, patch_size=32):
    """Returns a random "square-shaped" patch from a "sqare-shaped" image

    Args:
      image_size (int, optional): The width of the image. Defaults to 96.
      patch_size (int, optional): Width of the patch. Defaults to 32.
    """
    if patch_size > image_size:
        raise Exception("patch_size cannot be larger than image_size")

    # h_start = random.randint(0, image_size - patch_size)
    # v_start = random.randint(0, image_size - patch_size)

    # start_idx = v_start * image_size + h_start
    # first_row = np.arange(start_idx, start_idx + patch_size)

    horizontal_start = random.randint(1, image_size - (patch_size - 1))
    vertical_start = random.randint(1, image_size - (patch_size - 1))

    first_row = np.arange(
        (vertical_start - 1) * image_size + horizontal_start,
        (vertical_start - 1) * image_size + horizontal_start + patch_size,
    )
    
    patch_matrix = first_row + np.arange(patch_size)[:, np.newaxis] * image_size

    return patch_matrix.flatten()
  
# TODO: check if zero-indexing is required in ViT
# TODO: maybe prevent having patches on the edge?