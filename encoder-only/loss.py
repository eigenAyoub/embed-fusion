import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarityLoss(nn.Module):
    """Loss based on cosine similarity between all pairs in batch"""
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, model_output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        sim_outputs = nn.functional.cosine_similarity(model_output.unsqueeze(1), model_output.unsqueeze(0), dim=-1)
        sim_targets = nn.functional.cosine_similarity(targets.unsqueeze(1), targets.unsqueeze(0), dim=-1)

        mask = torch.eye(sim_outputs.size(0), device=sim_outputs.device).bool()
        sim_outputs = sim_outputs.masked_fill(mask, 0)
        sim_targets = sim_targets.masked_fill(mask, 0)
        
        loss = torch.mean((sim_outputs - sim_targets) ** 2)
        return self.weight * loss

class ImprovedSimilarityLoss(nn.Module):
    def __init__(self, weight: float = 1.0, margin: float = 0.1, temperature: float = 0.05):
        super().__init__()
        self.weight = weight
        self.margin = margin
        self.temperature = temperature

    def forward(self, model_output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute similarities
        sim_outputs = F.cosine_similarity(model_output.unsqueeze(1), 
                                        model_output.unsqueeze(0), dim=-1) / self.temperature
        sim_targets = F.cosine_similarity(targets.unsqueeze(1), 
                                        targets.unsqueeze(0), dim=-1) / self.temperature
        
        # Create mask for positive and negative pairs
        batch_size = sim_outputs.size(0)
        mask = torch.eye(batch_size, device=sim_outputs.device)
        
        # Compute loss with margin
        pos_loss = ((sim_outputs - sim_targets) ** 2) * mask
        neg_loss = torch.relu(sim_outputs - self.margin) * (1 - mask)
        
        loss = (pos_loss.sum() + neg_loss.sum()) / (batch_size * (batch_size - 1))
        return self.weight * loss

class SimilarityLossTopK(nn.Module):
    """Loss based on top-k cosine similarities between pairs in batch"""
    def __init__(self, weight: float = 1.0, k: int = 10):
        super().__init__()
        self.weight = weight
        self.k = k

    def forward(self, model_output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        sim_outputs = nn.functional.cosine_similarity(model_output.unsqueeze(1), model_output.unsqueeze(0), dim=-1)
        sim_targets = nn.functional.cosine_similarity(targets.unsqueeze(1), targets.unsqueeze(0), dim=-1)

        mask = torch.eye(sim_outputs.size(0), device=sim_outputs.device).bool()
        sim_outputs = sim_outputs.masked_fill(mask, -float('inf'))
        sim_targets = sim_targets.masked_fill(mask, -float('inf'))

        topk_values, topk_indices = sim_targets.topk(self.k, dim=-1)
        topk_sim_outputs = sim_outputs.gather(1, topk_indices)
        loss = torch.mean((topk_sim_outputs - topk_values) ** 2)
        return self.weight * loss
    

class ContrastiveInfoNCELoss(nn.Module):
    """
    An InfoNCE-style loss that encourages matching
    model_output[i] with targets[i] (positive) and
    treats all other pairs (i, j != i) as negatives.

    Args:
        temperature (float): Temperature to scale similarities.
        weight (float): Weight for the loss (if you want to combine with others).
    """
    def __init__(self, temperature: float = 0.05, weight: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.weight = weight

    def forward(self, model_output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            model_output: Tensor of shape (B, D), where B = batch size, D = embedding dimension.
            targets: Tensor of shape (B, D). Same shape as model_output.

        Returns:
            A scalar tensor representing the contrastive loss.
        """
        # Normalize for stable cosine similarity (optional but recommended)
        model_output = F.normalize(model_output, dim=-1)
        targets = F.normalize(targets, dim=-1)

        # Compute pairwise similarity (B x B)
        logits = torch.matmul(model_output, targets.transpose(0, 1))
        
        # Scale by temperature
        logits /= self.temperature

        # The label for each row is the "correct" index (the diagonal)
        labels = torch.arange(model_output.size(0), device=model_output.device)

        # Standard cross-entropy on the softmax over the batch dimension
        loss = F.cross_entropy(logits, labels)

        return self.weight * loss


