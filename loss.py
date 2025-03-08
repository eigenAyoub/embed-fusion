import torch
import torch.nn as nn
import torch.nn.functional as F

class Similarity(nn.Module):
    """Loss that penalizes the worst mismatches based on pairwise cosine similarity errors."""
    def __init__(self, weight: float = 1.0, k: int = 10):
        super().__init__()
        self.weight = weight
        self.k = k

    def forward(self, model_output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute pairwise cosine similarities
        sim_outputs = nn.functional.cosine_similarity(
            model_output.unsqueeze(1), model_output.unsqueeze(0), dim=-1
        )
        sim_targets = nn.functional.cosine_similarity(
            targets.unsqueeze(1), targets.unsqueeze(0), dim=-1
        )

        # Mask out self-similarities (set diagonal to 0 so they don't contribute to error)
        mask = torch.eye(sim_outputs.size(0), device=sim_outputs.device).bool()
        sim_outputs = sim_outputs.masked_fill(mask, 0.0)
        sim_targets = sim_targets.masked_fill(mask, 0.0)

        # Compute the squared error between predictions and targets for every pair
        error_matrix = (sim_outputs - sim_targets) ** 2

        # Select the worst k errors per sample
        worst_errors, _ = error_matrix.topk(self.k, dim=-1)

        # Aggregate the worst errors
        loss = worst_errors.mean()

        return self.weight * loss


class SimilarityLoss(nn.Module):
    """Loss based on cosine similarity between all pairs in batch"""
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, model_output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        sim_outputs = F.cosine_similarity(model_output.unsqueeze(1), model_output.unsqueeze(0), dim=-1)
        sim_targets = F.cosine_similarity(targets.unsqueeze(1), targets.unsqueeze(0), dim=-1)

        mask = torch.eye(sim_outputs.size(0), device=sim_outputs.device).bool()
        sim_outputs = sim_outputs.masked_fill(mask, 0)
        sim_targets = sim_targets.masked_fill(mask, 0)
        
        loss = torch.mean((sim_outputs - sim_targets) ** 2)
        return self.weight * loss

class KLSimilarityLoss(nn.Module):
    """KL Loss based on cosine similarity distributions between all pairs in a batch."""
    def __init__(self, weight: float = 1.0, temperature: float = 1.0):
        super().__init__()
        self.weight = weight
        self.temperature = temperature

    def forward(self, model_output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        sim_outputs = F.cosine_similarity(model_output.unsqueeze(1), model_output.unsqueeze(0), dim=-1)
        sim_targets = F.cosine_similarity(targets.unsqueeze(1), targets.unsqueeze(0), dim=-1)

        mask = torch.eye(sim_outputs.size(0), device=sim_outputs.device).bool()

        large_negative = -1e9
        sim_outputs = sim_outputs.masked_fill(mask, large_negative)
        sim_targets = sim_targets.masked_fill(mask, large_negative)
        
        student_log_probs = F.log_softmax(sim_outputs / self.temperature, dim=-1)
        teacher_probs     = F.softmax(sim_targets / self.temperature, dim=-1)
        
        loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        
        return self.weight * loss

class WeightedKLSimilarityLoss(nn.Module):
    def __init__(self, weight: float = 1.0, temperature: float = 1.0, power: float = 2.0):
        super().__init__()
        self.weight = weight
        self.temperature = temperature
        self.power = power

    def forward(self, model_output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        sim_outputs = F.cosine_similarity(model_output.unsqueeze(1), model_output.unsqueeze(0), dim=-1)
        sim_targets = F.cosine_similarity(targets.unsqueeze(1), targets.unsqueeze(0), dim=-1)

        mask = torch.eye(sim_outputs.size(0), device=sim_outputs.device).bool()
        sim_outputs = sim_outputs.masked_fill(mask, -1e9)
        sim_targets = sim_targets.masked_fill(mask, -1e9)

        student_log_probs = F.log_softmax(sim_outputs / self.temperature, dim=-1)
        teacher_probs = F.softmax(sim_targets / self.temperature, dim=-1)

        weights = (teacher_probs ** self.power).detach()
        loss = (weights * F.kl_div(student_log_probs, teacher_probs, reduction='none')).mean()

        return self.weight * loss

class InfoNCELoss(nn.Module):
    """InfoNCE Contrastive Loss based on cosine similarity."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        batch_size = embeddings1.size(0)

        sim_matrix = torch.mm(
            F.normalize(embeddings1, dim=-1), 
            F.normalize(embeddings2, dim=-1).T
        ) / self.temperature

        labels = torch.arange(batch_size, device=embeddings1.device)
        loss = F.cross_entropy(sim_matrix, labels)

        return loss
