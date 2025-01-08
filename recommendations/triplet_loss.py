import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Define a simple dataset
class TripletDataset(Dataset):
    def __init__(self, num_samples, feature_dim):
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Simulate feature vectors
        anchor = torch.rand(self.feature_dim)
        positive = anchor + 0.1 * torch.rand(self.feature_dim)  # Slightly similar to anchor
        negative = torch.rand(self.feature_dim)  # Completely random
        return anchor, positive, negative

# Define the embedding model
class EmbeddingModel(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, embedding_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1)

# Triplet loss with combined hard and semi-hard negative mining
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Compute distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        # Identify semi-hard negatives: d_pos < d_neg < d_pos + margin
        semi_hard_negatives = (neg_dist > pos_dist) & (neg_dist < (pos_dist + self.margin))

        # Identify too-hard negatives: d_neg < d_pos
        too_hard_negatives = neg_dist < pos_dist

        # Combine semi-hard and too-hard negatives
        selected_negatives = semi_hard_negatives | too_hard_negatives

        # Compute triplet loss for selected negatives
        loss = torch.clamp(pos_dist[selected_negatives] - neg_dist[selected_negatives] + self.margin, min=0.0)
        return loss.mean() if loss.numel() > 0 else torch.tensor(0.0, requires_grad=True)

# Training loop with hard and semi-hard negative mining
def train_triplet_model(num_epochs=5, batch_size=16, feature_dim=16, embedding_dim=8):
    # Initialize dataset and dataloader
    dataset = TripletDataset(num_samples=500, feature_dim=feature_dim)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and loss
    embedding_model = EmbeddingModel(input_dim=feature_dim, embedding_dim=embedding_dim)
    triplet_loss_fn = TripletLoss(margin=1.0)
    optimizer = torch.optim.Adam(embedding_model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            anchor, positive, negative = batch

            # Compute embeddings
            anchor_embed = embedding_model(anchor)
            positive_embed = embedding_model(positive)
            negative_embed = embedding_model(negative)

            # Compute triplet loss with hard and semi-hard negatives
            loss = triplet_loss_fn(anchor_embed, positive_embed, negative_embed)

            # Backward pass and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

# Run the training loop
train_triplet_model()
