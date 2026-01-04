# type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_distance(z):
    """Compute pairwise L2 distances between embeddings."""
    diff = z.unsqueeze(1) - z.unsqueeze(0)
    return torch.sum(diff ** 2, dim=-1)


class PairwiseSimilarity(nn.Module):
    def __init__(self, sigma=1.0):
        super(PairwiseSimilarity, self).__init__()
        self.sigma = sigma

    def forward(self, z):
        return torch.exp(-l2_distance(z) / self.sigma)


class ContextualSimilarity(nn.Module):
    def __init__(self, k):
        super(ContextualSimilarity, self).__init__()
        self.k = k

    def forward(self, z):
        distances = l2_distance(z)
        # Get k-th nearest neighbor distances (using negative for top_k)
        kth_nearest = torch.topk(-distances, k=self.k, dim=-1, sorted=True)[0][:, -1]
        mask = (distances <= kth_nearest.unsqueeze(-1)).float()

        similarity = torch.matmul(mask, mask.t()) / torch.sum(mask, dim=-1, keepdim=True)
        R = mask * mask.t()
        similarity = torch.matmul(similarity, R.t()) / torch.sum(R, dim=-1, keepdim=True)
        return 0.5 * (similarity + similarity.t())


class ReConPatch(nn.Module):
    def __init__(self, input_dim, embedding_dim, projection_dim, alpha, margin=0.1, ema_decay=0.9):
        super(ReConPatch, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.ema_decay = ema_decay

        # Embedding & projection layers
        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.projection = nn.Linear(embedding_dim, projection_dim)

        # EMA versions of embedding & projection layers
        self.ema_embedding = nn.Linear(input_dim, embedding_dim)
        self.ema_projection = nn.Linear(embedding_dim, projection_dim)

        # Disable gradients for EMA layers
        for param in self.ema_embedding.parameters():
            param.requires_grad = False
        for param in self.ema_projection.parameters():
            param.requires_grad = False

        # Initialize EMA layers with same weights
        self.update_ema(init=True)

        self.pairwise_similarity = PairwiseSimilarity(sigma=1.0)
        self.contextual_similarity = ContextualSimilarity(k=3)

    def forward(self, x):
        return self.embedding(x)

    def train_step(self, x, optimizer):
        """
        Perform a single training step.

        Args:
            x: Input tensor
            optimizer: PyTorch optimizer

        Returns:
            Dictionary containing the loss value
        """
        # Forward pass through EMA networks (no gradient)
        with torch.no_grad():
            h_ema = self.ema_embedding(x)
            z_ema = self.ema_projection(h_ema)

            p_sim = self.pairwise_similarity(z_ema)
            c_sim = self.contextual_similarity(z_ema)
            w = self.alpha * p_sim + (1 - self.alpha) * c_sim

        # Forward pass through trainable networks
        h = self.embedding(x)
        z = self.projection(h)

        # Contrastive loss
        distances = torch.sqrt(l2_distance(z) + 1e-9)
        delta = distances / torch.mean(distances, dim=-1, keepdim=True)
        rc_loss = torch.sum(torch.mean(
            w * (delta ** 2) + (1 - w) * (F.relu(self.margin - delta) ** 2),
            dim=-1
        ))

        # Update weights
        optimizer.zero_grad()
        rc_loss.backward()
        optimizer.step()

        # Update EMA
        self.update_ema()

        return {"rc_loss": rc_loss.item()}

    @torch.no_grad()
    def update_ema(self, init=False):
        """Update EMA weights using exponential moving average."""
        if init:
            # Initialize EMA weights with current weights
            self.ema_embedding.weight.data.copy_(self.embedding.weight.data)
            self.ema_embedding.bias.data.copy_(self.embedding.bias.data)
            self.ema_projection.weight.data.copy_(self.projection.weight.data)
            self.ema_projection.bias.data.copy_(self.projection.bias.data)
        else:
            # Update EMA weights
            for ema_param, param in zip(self.ema_embedding.parameters(), self.embedding.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
            for ema_param, param in zip(self.ema_projection.parameters(), self.projection.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
