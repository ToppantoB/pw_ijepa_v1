from .knn_eval import get_knn_accuracy
from .linear_probe_eval import get_linear_probe_accuracy

from config import CONFIG
from utils import get_project_root

import os


def get_path(run_version, param_version, save_type):
    """Constructs and returns the file path for model outputs.
       Designed to work with the format the training cycle saves the models.

    Args:
      run_version (int or str): The version identifier for the current run.
      param_version (int or str): The version identifier for the model parameters.
      save_type (str): The specific save category (e.g., "best" or "latest").

    Returns:
      str: The resolved file path, prioritizing the "OUT_DIR" environment variable if set.
    """
    project_root = get_project_root()
    return project_root / f"outputs/v{run_version}/{save_type}/ijepa_stl10_{param_version}.pt"
  


def do_eval(
    run_version=None,
    param_version=None,
    encoder_config={"depth": CONFIG.encoder_depth, "patch_size": CONFIG.patch_size},
    save_type="best",
    eval_type="knn",
    path=None,
    show_logs=False,
    device="cuda"
):
    """Evaluates a saved model using either K-Nearest Neighbors or a linear probe.

    Args:
      run_version (int or str, optional): The version identifier for the run. Defaults to None.
      param_version (int or str, optional): The version identifier for the parameters. Defaults to None.
      encoder_config (dict, optional): Configuration parameters for the encoder. Defaults to {"depth": CONFIG.encoder_depth, "patch_size": CONFIG.patch_size}.
      save_type (str, optional): The category of the saved model to load. Defaults to "best".
      eval_type (str, optional): The evaluation method to use ("knn" or "linear_probe"). Defaults to "knn".
      path (str, optional): Explicit file path to the model. If None, the path is constructed automatically. Defaults to None.
      show_logs (bool, optional): Controls the verbosity of both the training and evaluation steps. Defaults to False.
      device (str or torch.device): The device (e.g., 'cuda' or 'cpu') to perform computations on. Defaults to "cuda"

    Returns:
      float: The evaluation accuracy score resulting from the chosen evaluation method.
    """
    if path is None:
        path = get_path(run_version, param_version, save_type)

    if eval_type == "knn":
        return get_knn_accuracy(path, encoder_config, device)
    elif eval_type == "linear_probe":
        return get_linear_probe_accuracy(path, encoder_config, device, show_logs)
