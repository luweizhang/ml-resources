import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Define dataset for user-item interactions
class TripletIDDataset(Dataset):
    def __init__(self, num_users, num_videos, num_samples):
        self.num_users = num_users
        self.num_videos = num_videos
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Randomly generate user and video IDs
        user_id = torch.randint(0, self.num_users, (1,))
        pos_video_id = torch.randint(0, self.num_videos, (1,))
        neg_video_id = torch.randint(0, self.num_videos, (1,))
        return user_id, pos_video_id, neg_video_id

# Define the Two-Tower model with FFN
class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_videos, embedding_dim, hidden_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.video_embedding = nn.Embedding(num_videos, embedding_dim)

        # Feedforward layers for user tower
        self.user_ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)  # Project back to embedding size
        )

        # Feedforward layers for video tower
        self.video_ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)  # Project back to embedding size
        )

    def forward(self, user_id, video_id):
        user_embed = self.user_embedding(user_id)
        video_embed = self.video_embedding(video_id)

        # Pass through feedforward networks
        user_embed = self.user_ffn(user_embed)
        video_embed = self.video_ffn(video_embed)

        return user_embed, video_embed

# Define Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, pos, neg):
        pos_dist = F.pairwise_distance(anchor, pos, p=2)
        neg_dist = F.pairwise_distance(anchor, neg, p=2)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()

# Training loop
def train_triplet_model(num_users=10000, num_videos=100000, embedding_dim=128, hidden_dim=256, num_epochs=5, batch_size=16):
    dataset = TripletIDDataset(num_users, num_videos, num_samples=5000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TwoTowerModel(num_users, num_videos, embedding_dim, hidden_dim)
    loss_fn = TripletLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for user_id, pos_video_id, neg_video_id in dataloader:
            user_embed, pos_video_embed = model(user_id, pos_video_id)
            _, neg_video_embed = model(user_id, neg_video_id)

            loss = loss_fn(user_embed, pos_video_embed, neg_video_embed)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

# Run training
train_triplet_model()
