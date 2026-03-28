import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import os

from transformer import VisionTransformer
from wandb_cfg import run
from test_model import test_model

transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

DATA_ROOT = os.getenv("DATA_ROOT", "./data")
OUT_DIR = os.getenv("OUT_DIR", "./outputs")
os.makedirs(OUT_DIR, exist_ok=True)

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

epochs = 3
warmup_epochs = 1
steps_per_epoch = len(train_loader)
total_steps = steps_per_epoch * epochs
warmup_steps = steps_per_epoch * warmup_epochs

model = VisionTransformer().to("cuda")
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

# 1. Warmup Phase: Start at 1% of the base LR (3e-6) and scale up to 100% (3e-4)
warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)

# 2. Decay Phase: Smoothly curve back down to 0 over the remaining steps
decay_scheduler = CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps))

# 3. Combine them into a single scheduler
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_steps])


best_accuracy, best_path = 0.0, os.path.join(OUT_DIR, "cifar_vit.pt")


for epoch in range(epochs):
  model.train()
  running_loss = 0.0
  
  pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
  
  for inputs, labels in pbar:
    inputs, labels = inputs.cuda(), labels.cuda()
    optimizer.zero_grad()
    
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    running_loss += loss.item()
    
  train_loss = running_loss / len(train_loader)
  
  print(f"\nEvaluating Epoch {epoch+1}...")
  val_loss, val_accuracy = test_model(model, test_loader, criterion)
    
  if val_accuracy > best_accuracy:
    best_accuracy = val_accuracy
    torch.save(model.state_dict(), best_path)
    print(f"Saved: {best_path} (acc={best_accuracy:.4f})")
    
  run.log({
        "loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })
  
run.finish()