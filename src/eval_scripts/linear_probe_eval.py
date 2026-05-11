import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from .test_models import get_encoder
from .test_loaders import train_labeled_loader as train_loader, test_loader


def get_linear_head(encoder, device, show_training_logs=False):
    """Trains and returns a linear classification head on top of a frozen encoder.

    Args:
      encoder (torch.nn.Module): The pre-trained neural network encoder.
      device (str or torch.device): The device (e.g., 'cuda' or 'cpu') to perform computations on.
      show_training_logs (bool, optional): Whether to display a progress bar and print statements during training. Defaults to False.

    Returns:
      torch.nn.Linear: The trained linear classification probe.
    """
    epochs = 100
    if not show_training_logs:
        print("Linear probe training in progress...")

    for param in encoder.parameters():
        param.requires_grad = False

    encoder.eval()

    linear_probe = nn.Linear(192, 10).to(device)
    linear_probe.train()

    optimizer = torch.optim.AdamW(linear_probe.parameters(), lr=1e-3, weight_decay=0.01)

    epoch_iterator = tqdm(range(epochs), desc="Training Linear Probe") if show_training_logs else range(epochs)

    for _ in epoch_iterator:
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                features = encoder(images)
                if features.dim() == 3:
                    features = features.mean(dim=1)

            logits = linear_probe(features)
            loss = nn.functional.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return linear_probe


def linear_probe_eval(linear_head, encoder, device):
    """Evaluates the accuracy of a linear probe and encoder on the test dataset.

    Args:
      linear_head (torch.nn.Module): The trained linear classification head.
      encoder (torch.nn.Module): The pre-trained neural network encoder.

    Returns:
      float: The final linear probing accuracy as a percentage.
    """
    encoder.eval()
    linear_head.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            features = encoder(images)
            if features.dim() == 3:
                features = features.mean(dim=1)

            logits = linear_head(features)
            predictions = torch.argmax(logits, dim=-1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    final_accuracy = (correct / total) * 100
    print(f"Linear Probing Accuracy: {final_accuracy:.2f}%")
    return final_accuracy

def get_linear_probe_accuracy(path_to_params, encoder_config, device, show_logs=False):
    """Loads an encoder, trains a linear head, and computes its evaluation accuracy.

    Args:
      path_to_params (str): The file path to the saved model parameters.
      encoder_config (dict): Configuration parameters required to initialize the encoder.
      device (str or torch.device): The device (e.g., 'cuda' or 'cpu') to perform computations on.
      show_logs (bool, optional): Controls the verbosity of both the training and evaluation steps. Defaults to False.

    Returns:
      float: The computed linear probing accuracy across the test dataset.
    """
    context_encoder = get_encoder(path_to_params, encoder_config, device)

    linear_head = get_linear_head(context_encoder, device, show_training_logs=show_logs)
    return linear_probe_eval(linear_head, context_encoder, device)
