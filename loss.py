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
        # Compute cosine similarity matrices for student and teacher
        sim_outputs = F.cosine_similarity(model_output.unsqueeze(1), model_output.unsqueeze(0), dim=-1)
        sim_targets = F.cosine_similarity(targets.unsqueeze(1), targets.unsqueeze(0), dim=-1)

        mask = torch.eye(sim_outputs.size(0), device=sim_outputs.device).bool()
        sim_outputs = sim_outputs.masked_fill(mask, float('-inf'))
        sim_targets = sim_targets.masked_fill(mask, float('-inf'))
        
        # Apply softmax (and log_softmax for the student) with temperature scaling
        student_log_probs = F.log_softmax(sim_outputs / self.temperature, dim=-1)
        teacher_probs     = F.softmax(sim_targets / self.temperature, dim=-1)
        
        # Compute the KL divergence for each row and average over the batch
        loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        
        return self.weight * loss




    