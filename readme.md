
## Description

This project implements a scaled down version of the I-JEPA model for the *Practical work in AI* course at the *Johannes Kepler University, Linz*. The implementation is based on the original [I-JEPA paper by Assran et al. (2024)](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html) and is designed to be trained on the STL-10 dataset. The code is structured to allow for easy training and evaluation of the model on consumer-grade hardware.

## Installation

1. Clone the repository:
  ```bash
  git clone https://github.com/ToppantoB/pw_ijepa_v1.git
  cd pw_ijepa_v1
  ```

2. Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Usage
#### Training

Train the model with the command:
```bash
python main.py
```
This starts a training cycle according to the configuration that can be found in the `config.py` file. Training checkpoints are saved periodically to the `outputs` folder. See more at the [Configuration](#configuration) section.

#### Evaluation
You can evaluate a saved model with the following command:
```bash
python main.py --eval --modelPath path/to/your/model.pt
```
This will train a linear probe on top of the frozen encoder and evaluate its performance on the STL-10 test set.
There is a model checkpoint included in the `saved_models` folder that you can use for evaluation. The checkpoint was trained for 220 epochs and achieves a linear probe accuracy of around 70% on the STL-10 test set. To evaluate this checkpoint, run the following command:
```bash
python main.py --eval --modelPath saved_models/baseline.pt
```


## Configuration
The model and training parameters can be configured in the `config.py` file. To better understand the configuration options, you can find an illustration of the I-JEPA architecture [below](#ijepa_image). The most relevant parameters are:
##### Image and patch related:
- `patch_size`: Size of the image patches (default: 8 to break the 96x96 STL-10 images into 144 patches).
- `block_size`: Size of the blocks in patches that are masked during training (default: 4 to mask 4x4 blocks of patches).
- `number_of_blocks`: Number of blocks to mask during training (default: 5).
- `min_context_size`: In I-JEPA the image for the context encoder is randomly cropped before the target blocks are subtracted. This parameter sets the minimum size of the image in patches after cropping. (default: 10 to ensure that before the target blocks are subtracted, there is enough context available).

##### Training related:
- `epochs`: Number of training epochs (default: 220).
- `batch_size`: Batch size for training (default: 128).
- `grad_accum_steps`: Number of steps to accumulate gradients before performing an optimizer step (default: 2, meaning an effective batch size of 256).
- `base_learning_rate`: Base learning rate for the optimizer (default: 1.5e-4).
- `tau_base`: Base tau for the EMA update of the target encoder (default: 0.985).
- `tau_end`: Final tau for the EMA update of the target encoder (default: 1).
- `predictor_lr_multiplier`: Multiplier for the learning rate of the predictor head (default: 2).

##### Encoder related:
- `encoder_embed_dim`: Embedding dimension for the encoder (default: 192).
- `encoder_num_heads`: Number of attention heads in the encoder (default: 3).
- `encoder_depth`: Number of transformer layers in the encoder (default: 12).
- `encoder_mlp_dim`: Dimension of the MLP in the transformer layers (default: 768).

##### Predictor related:
- `predictor_embed_dim`: Embedding dimension for the predictor head (default: 192).
- `predictor_num_heads`: Number of attention heads in the predictor head (default: 3).
- `predictor_depth`: Number of transformer layers in the predictor head (default: 3).
- `predictor_mlp_dim`: Dimension of the MLP in the predictor head (default: 768).

<img src="./readme_images/ijepa_main.png" alt="Logo" width="600" id="ijepa_image">


## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt` (install via pip as above)
- CUDA enabled GPU for training and inference (optional but recommended)
