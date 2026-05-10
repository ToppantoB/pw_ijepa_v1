import torch
import numpy as np
import torch.nn.functional as F

from config import CONFIG
from .test_loaders import train_labeled_loader, test_loader
from .test_models import get_encoder

@torch.no_grad()
def extract_features(dataloader, encoder, device):
    """Extracts and pools features for the entire dataset."""
    encoder.eval()
    features_list = []
    labels_list = []
    
    for images, labels in dataloader:
        images = images.to(device)
        
        out = encoder(images)
        
        if out.dim() == 3:
            out = out.mean(dim=1)
            
        features_list.append(out.cpu())
        labels_list.append(labels.cpu())
        
    return torch.cat(features_list), torch.cat(labels_list)

def knn_evaluate(train_features, train_labels, test_features, test_labels, k=20):
    """Performs k-NN classification using cosine similarity."""
    
    train_features = F.normalize(train_features, dim=1)
    test_features = F.normalize(test_features, dim=1)
    
    similarity_matrix = torch.mm(test_features, train_features.t())
    
    _, topk_indices = similarity_matrix.topk(k, dim=1)
    
    topk_labels = train_labels[topk_indices]
    
    predictions = topk_labels.mode(dim=1).values
    
    accuracy = (predictions == test_labels).float().mean().item()
    
    print(f"k-NN accuracy measured at: {accuracy}")
    
    return accuracy

def get_knn_accuracy(path_to_params, encoder_config):
  context_encoder = get_encoder(path_to_params, encoder_config)

  train_features, train_labels = extract_features(train_labeled_loader, context_encoder, "cuda")
  test_features, test_labels = extract_features(test_loader, context_encoder, "cuda")

  acc = knn_evaluate(train_features, train_labels, test_features, test_labels)

  return acc