import torch
import torch.nn as nn
import numpy as np

from .test_models import get_encoder
from .test_loaders import train_labeled_loader as train_loader, test_loader


def get_linear_head(encoder):
    epochs = 100
    print("Linear probe eval in progress...")

    for param in encoder.parameters():
        param.requires_grad = False

    encoder.eval()

    linear_probe = nn.Linear(192, 10).to("cuda")
    linear_probe.train()

    optimizer = torch.optim.AdamW(linear_probe.parameters(), lr=1e-3, weight_decay=0.01)

    for _ in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to("cuda"), labels.to("cuda")

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


def linear_probe_eval(linear_head, encoder):
    encoder.eval()
    linear_head.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to("cuda"), labels.to("cuda")

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

def get_linear_probe_accuracy(path_to_params, encoder_config):
    context_encoder = get_encoder(path_to_params, encoder_config)

    linear_head = get_linear_head(context_encoder)
    return linear_probe_eval(linear_head, context_encoder)
