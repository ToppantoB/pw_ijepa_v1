import torchvision
import torch
from torchvision import transforms

from config import CONFIG
from utils import get_project_root

def get_train_loader():
    """Creates and returns a DataLoader for the STL10 training dataset with specified transformations.
    
    Returns:
      torch.utils.data.DataLoader: A DataLoader object for the STL10 training dataset.
    """
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

    data_path = get_project_root() / "data"

    dataset = torchvision.datasets.STL10(
        root=data_path, split="unlabeled", download=True, transform=transform
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=CONFIG.batch_size, drop_last=True
    )
