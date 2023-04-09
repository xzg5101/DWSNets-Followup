import torch
import torch.nn.functional as F

def info_nce_loss(features, temperature):
    """
    Optimized by chatGpt based on Source: https://github.com/sthalles/SimCLR/blob/master/simclr.py

    :param features:
    :param temperature:
    :return:
    """
    n_views = 2
    bs = features.shape[0] // n_views
    device = features.device

    # Create labels
    labels = torch.arange(bs, device=device).repeat(n_views)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    # Normalize features
    features = F.normalize(features, dim=1)

    # Calculate similarity matrix using einsum for better performance
    similarity_matrix = torch.einsum('ik,jk->ij', features, features)

    # Remove diagonals from labels and similarity_matrix
    mask = torch.eye(bs * n_views, dtype=torch.bool, device=device)
    labels = labels.masked_select(~mask).view(bs * n_views, -1)
    similarity_matrix = similarity_matrix.masked_select(~mask).view(bs * n_views, -1)

    # Extract positives and negatives
    positives = similarity_matrix[labels.bool()].view(bs * n_views, -1)
    negatives = similarity_matrix[~labels.bool()].view(bs * n_views, -1)

    # Concatenate logits and create ground truth labels
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(bs * n_views, dtype=torch.long, device=device)

    # Apply temperature
    logits /= temperature

    return logits, labels
