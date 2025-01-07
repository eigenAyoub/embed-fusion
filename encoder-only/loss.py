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



class PairwiseCosineMarginLoss(nn.Module):
    """
    A margin-based cosine similarity loss that handles the case where
    `model_output` is (B, D1) but `targets` might be (B*N, D2).

    For each sample i in model_output:
      - We assume there are N "positive" targets in `targets`
        (i.e. i*N through i*N+N-1 if we chunk `targets` appropriately).
      - All other elements in `targets` are "negatives."

    Cosine similarity is computed between each anchor and each target.
    Positives are pushed to high similarity (close to 1),
    negatives are pushed below the margin.

    Args:
        margin (float): Margin for negative pairs. If sim > margin, we penalize.
        weight (float): Optional weight multiplier for the loss.
    """
    def __init__(self, margin: float = 0.1, weight: float = 1.0):
        super().__init__()
        self.margin = margin
        self.weight = weight

    def forward(self, model_output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            model_output: (B, D1)
            targets: (B*N, D2)  for some integer N >= 1

        Returns:
            A scalar tensor (the loss).
        """
        B, D1 = model_output.shape
        BT, D2 = targets.shape

        # 1) Check that BT is a multiple of B
        if BT % B != 0:
            raise ValueError(
                f"targets.size(0)={BT} is not a multiple of model_output.size(0)={B}."
            )
        N = BT // B

        # 2) If embedding dimensions differ, project or raise an error
        if D1 != D2:
            raise ValueError(
                f"Embedding dimension mismatch: D1={D1}, D2={D2}. "
                "Either project them to the same size or fix your model/targets."
            )

        # 3) Reshape targets so that for each i in [0..B-1],
        #    targets[i] => chunk [i*N .. i*N+N-1]
        #    => shape: (B, N, D2)
        targets = targets.view(B, N, D2)

        # 4) Normalize embeddings (optional but recommended for cosine-based losses)
        model_output = F.normalize(model_output, dim=-1)  # (B, D)
        targets = F.normalize(targets, dim=-1)            # (B, N, D)

        # 5) Compute pairwise similarity:
        #    For each i in [0..B-1], compare model_output[i] with EVERY target in [0..B-1, 0..N-1]
        #    We'll get a (B, B*N) similarity matrix
        #
        #    Approach: 
        #    - Expand model_output from (B, D) -> (B, 1, D) => (B, N, D) if we want just i vs i*N.. 
        #      but we also want i vs all, so let's do (B, D) x (B*N, D) => (B, B*N).
        
        # Flatten targets again as (B*N, D) for convenience:
        targets_flat = targets.view(B*N, D2)  # (B*N, D)

        # => (B, D) @ (D, B*N) -> (B, B*N)
        # But we do the dot-product ourselves:
        sim_matrix = torch.matmul(model_output, targets_flat.transpose(0, 1))  # (B, B*N)

        # 6) Identify positive pairs vs negative pairs via a mask
        #    We want a mask of shape (B, B*N)
        #    For anchor i in [0..B-1], positives are j in [i*N..(i+1)*N - 1].
        #    So let's build an index array: 
        #      row i => range(i*N, i*N+N)
        # We'll do a "gather" approach or direct boolean mask.

        # Create a (B*N,) label array that says which anchor each target belongs to:
        # e.g. target_indices = [0]*N + [1]*N + ... + [B-1]*N
        target_owner = torch.arange(B, device=sim_matrix.device).repeat_interleave(N)  # shape (B*N)

        # Now each row i in sim_matrix should be "positive" with columns where target_owner == i
        # We'll build a boolean mask:
        # pos_mask[i, j] = (target_owner[j] == i)
        pos_mask = (target_owner.unsqueeze(0) == torch.arange(B, device=sim_matrix.device).unsqueeze(1))
        # shape of pos_mask => (B, B*N)
        # row i => True where target_owner == i, else False

        # 7) We want positives => high similarity, negatives => below margin
        #   pos_loss = 1 - sim for positive entries
        #   neg_loss = relu(sim - margin) for negative entries
        # Because sim_matrix is (B, B*N), we do elementwise on the mask:
        pos_sim = sim_matrix[pos_mask]        # 1D
        neg_sim = sim_matrix[~pos_mask]       # 1D

        pos_loss = (1.0 - pos_sim).clamp_min(0)   # push positives near 1
        neg_loss = F.relu(neg_sim - self.margin)  # push negatives < margin

        # 8) Combine losses
        #    total number of positives = B * N
        #    total number of negatives = B * B*N - B*N = B*N*(B-1)
        # We'll just average over everything to keep it simple
        loss = (pos_loss.sum() + neg_loss.sum()) / (pos_loss.numel() + neg_loss.numel())

        return self.weight * loss
