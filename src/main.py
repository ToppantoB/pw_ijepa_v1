import argparse

from train import train
from utils import get_compute_device


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handle conditional arguments.")

    parser.add_argument("--eval", action="store_true", help="Enable evaluation mode")
    parser.add_argument("--modelPath", type=str, help="Path to the model (required if --eval is used)")
    
    args = parser.parse_args()
    
    device = get_compute_device()
    
    if args.eval:
        if args.modelPath is None:
            parser.error("--modelPath is required when --eval is specified.")
            
        # only import if needed  
        from eval_scripts.eval import do_eval

        do_eval(eval_type="linear_probe", path=args.modelPath, show_logs=True, device=device)
    else: 
        train(device)
    
