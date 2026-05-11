from torchvision import datasets, transforms
import torch

from config import CONFIG
from utils import get_project_root


eval_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713]
        ),
    ]
)

data_path = get_project_root() / "data"

train_labeled_ds = datasets.STL10(
    root=data_path, split="train", transform=eval_transform, download=True
)

test_ds = datasets.STL10(root=data_path, split="test", transform=eval_transform)

# initialize the train and test loaders for the linear probe evaluation
train_labeled_loader = torch.utils.data.DataLoader(train_labeled_ds, batch_size=CONFIG.batch_size)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=CONFIG.batch_size)