import torch
import numpy as np
import torch.nn.functional as F

from config import CONFIG
from .test_loaders import train_labeled_loader, test_loader
from .test_models import get_encoder

@torch.no_grad()
def extract_features(dataloader, encoder, device):
    """Extracts and pools features for the entire dataset.

    Args:
      dataloader (torch.utils.data.DataLoader): The data loader providing images and labels.
      encoder (torch.nn.Module): The neural network encoder model used to extract features.
      device (str or torch.device): The device (e.g., 'cuda' or 'cpu') to perform computations on.

    Returns:
      tuple: A tuple containing two torch.Tensor objects representing the concatenated features and labels.
    """
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
    """Performs k-NN classification using cosine similarity.

    Args:
      train_features (torch.Tensor): The extracted features from the training dataset.
      train_labels (torch.Tensor): The corresponding labels for the training features.
      test_features (torch.Tensor): The extracted features from the test dataset.
      test_labels (torch.Tensor): The corresponding labels for the test features.
      k (int, optional): The number of nearest neighbors to consider. Defaults to 20.

    Returns:
      float: The measured k-NN accuracy as a float between 0.0 and 1.0.
    """
    
    train_features = F.normalize(train_features, dim=1)
    test_features = F.normalize(test_features, dim=1)
    
    similarity_matrix = torch.mm(test_features, train_features.t())
    
    _, topk_indices = similarity_matrix.topk(k, dim=1)
    
    topk_labels = train_labels[topk_indices]
    
    predictions = topk_labels.mode(dim=1).values
    
    accuracy = (predictions == test_labels).float().mean().item()
    
    print(f"k-NN accuracy measured at: {accuracy}")
    
    return accuracy

def get_knn_accuracy(path_to_params, encoder_config, device):
  """Loads an encoder model and computes its k-NN evaluation accuracy.

  Args:
      path_to_params (str): The file path to the saved model parameters.
      encoder_config (dict): Configuration parameters required to initialize the encoder.
      device (str or torch.device): The device (e.g., 'cuda' or 'cpu') to perform computations on.
  
  Returns:
      float: The computed k-NN accuracy across the test dataset.
  """
  context_encoder = get_encoder(path_to_params, encoder_config, device)

  train_features, train_labels = extract_features(train_labeled_loader, context_encoder, device)
  test_features, test_labels = extract_features(test_loader, context_encoder, device)

  acc = knn_evaluate(train_features, train_labels, test_features, test_labels)

  return acc