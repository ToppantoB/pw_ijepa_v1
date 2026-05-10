import torchvision
import torch
from torchvision import transforms
import os

from config import CONFIG


def get_train_loader():
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(96, (0.2, 1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713]
            ),
        ]
    )

    data_path = os.path.abspath("./data")

    dataset = torchvision.datasets.STL10(
        root=data_path, split="unlabeled", download=True, transform=transform
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=CONFIG.batch_size, drop_last=True
    )
