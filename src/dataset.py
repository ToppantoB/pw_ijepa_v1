import torchvision
import torch
from torchvision import transforms
import os

from config import CONFIG


def get_train_loader():
    transform = transforms.Compose(
        [
            # transforms.RandomCrop(96, padding=4, padding_mode="reflect"),
            transforms.RandomResizedCrop(96, (0.2, 1)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.2),
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
    # return torch.utils.data.DataLoader(dataset, batch_size=CONFIG.batch_size, num_workers=2)
