from .knn_eval import get_knn_accuracy
from .linear_probe_eval import get_linear_probe_accuracy

from config import CONFIG

import os


def get_path(run_version, param_version, save_type):
    return os.getenv(
        "OUT_DIR",
        f"./outputs/v{run_version}/{save_type}/ijepa_stl10_{param_version}.pt",
    )


def do_eval(
    run_version,
    param_version,
    encoder_config={"depth": CONFIG.encoder_depth, "patch_size": CONFIG.patch_size},
    save_type="best",
    eval_type="knn",
):
    path = get_path(run_version, param_version, save_type)

    if eval_type == "knn":
        return get_knn_accuracy(path, encoder_config)
    elif eval_type == "linear_probe":
        return get_linear_probe_accuracy(path, encoder_config)
