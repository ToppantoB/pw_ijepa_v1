import torchvision

# Download the 100,000 unlabeled images for I-JEPA pre-training
dataset = torchvision.datasets.STL10(
    root='./data', 
    split='unlabeled', 
    download=True
)