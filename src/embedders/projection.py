"""
Projection layers for embedding dimension manipulation.

Includes:
- ProjectionLayer: Simple linear projection (512D -> 1024D)
- MatryoshkaProjection: Variable dimension output with nested embeddings (512D -> 256D/128D/64D)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import List, Optional, Union, Tuple
from enum import Enum


class MatryoshkaDimension(Enum):
    """Supported Matryoshka embedding dimensions."""
    DIM_512 = 512
    DIM_256 = 256
    DIM_128 = 128
    DIM_64 = 64
    DIM_32 = 32  # Ultra-low for very constrained devices


# Default dimensions in descending order (largest first)
DEFAULT_MATRYOSHKA_DIMS = [512, 256, 128, 64]


class ProjectionLayer(nn.Module):
    """Linear projection layer with normalization."""

    def __init__(self, input_dim: int = 512, output_dim: int = 1024):
        """
        Initialize projection layer.

        Args:
            input_dim: Input embedding dimension
            output_dim: Output embedding dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings to higher dimension with normalization.

        Args:
            x: Input embeddings (shape: [batch_size, input_dim] or [input_dim])

        Returns:
            Projected and normalized embeddings
        """
        projected = self.linear(x)
        # L2 normalization
        normalized = projected / projected.norm(dim=-1, keepdim=True)
        return normalized

    def save(self, path: str):
        """Save projection layer weights."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim
        }, path)
        print(f"Projection layer saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        """Load projection layer from disk."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            input_dim=checkpoint['input_dim'],
            output_dim=checkpoint['output_dim']
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        print(f"Projection layer loaded from {path}")
        return model


class MatryoshkaProjection(nn.Module):
    """
    Matryoshka Representation Learning projection layer.

    Produces embeddings that can be truncated to smaller dimensions while
    preserving semantic quality. This is achieved by training with a
    multi-scale loss that ensures the first N dimensions are meaningful.

    Key features:
    - Variable output dimensions (512, 256, 128, 64, 32)
    - Nested embeddings: dim_64 ⊂ dim_128 ⊂ dim_256 ⊂ dim_512
    - Mobile-friendly: Use smaller dims on constrained devices
    - Compatible with CLIP embeddings as input

    Reference: Matryoshka Representation Learning (Kusupati et al., 2022)
    https://arxiv.org/abs/2205.13147
    """

    def __init__(
        self,
        input_dim: int = 512,
        matryoshka_dims: List[int] = None,
        use_mlp: bool = True,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Initialize Matryoshka projection layer.

        Args:
            input_dim: Input embedding dimension (e.g., 512 from CLIP)
            matryoshka_dims: List of output dimensions in descending order
                             Default: [512, 256, 128, 64]
            use_mlp: Use MLP projection instead of linear (better quality)
            hidden_dim: Hidden dimension for MLP (default: input_dim * 2)
            dropout: Dropout rate for regularization
        """
        super().__init__()

        self.input_dim = input_dim
        self.matryoshka_dims = matryoshka_dims or DEFAULT_MATRYOSHKA_DIMS
        self.max_dim = max(self.matryoshka_dims)
        self.use_mlp = use_mlp
        self.dropout = dropout

        # Validate dimensions
        assert all(d <= input_dim for d in self.matryoshka_dims), \
            f"All matryoshka dims must be <= input_dim ({input_dim})"

        # Sort dimensions in descending order
        self.matryoshka_dims = sorted(self.matryoshka_dims, reverse=True)

        # Build projection network
        hidden_dim = hidden_dim or input_dim * 2

        if use_mlp:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.max_dim),
            )
        else:
            self.projection = nn.Linear(input_dim, self.max_dim)

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(self.max_dim)

        # Dimension-specific scaling factors (learned)
        # These help each truncated dimension maintain good properties
        self.dim_scales = nn.ParameterDict({
            str(dim): nn.Parameter(torch.ones(1))
            for dim in self.matryoshka_dims
        })

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better convergence."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        output_dim: Optional[int] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Project embeddings with optional dimension truncation.

        Args:
            x: Input embeddings [batch_size, input_dim] or [input_dim]
            output_dim: Desired output dimension. If None, returns max_dim.
                        Must be one of matryoshka_dims.
            normalize: Apply L2 normalization (default: True)

        Returns:
            Projected embeddings [batch_size, output_dim] or [output_dim]
        """
        # Handle 1D input
        squeeze_output = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True

        # Project to max dimension
        projected = self.projection(x)
        projected = self.layer_norm(projected)

        # Truncate if requested
        if output_dim is not None:
            if output_dim not in self.matryoshka_dims:
                # Find nearest supported dimension
                nearest = min(self.matryoshka_dims, key=lambda d: abs(d - output_dim))
                output_dim = nearest

            projected = projected[:, :output_dim]

            # Apply dimension-specific scaling
            scale = self.dim_scales[str(output_dim)]
            projected = projected * scale

        # L2 normalize
        if normalize:
            projected = F.normalize(projected, p=2, dim=-1)

        if squeeze_output:
            projected = projected.squeeze(0)

        return projected

    def forward_multi_scale(
        self,
        x: torch.Tensor,
        dims: Optional[List[int]] = None,
        normalize: bool = True
    ) -> dict:
        """
        Get embeddings at multiple scales simultaneously.

        Useful for:
        - Training with multi-scale loss
        - Comparing representations at different granularities

        Args:
            x: Input embeddings [batch_size, input_dim]
            dims: List of dimensions to return (default: all matryoshka_dims)
            normalize: Apply L2 normalization

        Returns:
            Dictionary mapping dimension -> embeddings
        """
        dims = dims or self.matryoshka_dims

        # Get full projection once
        full_proj = self.forward(x, output_dim=None, normalize=False)

        results = {}
        for dim in dims:
            if dim > self.max_dim:
                continue
            truncated = full_proj[:, :dim]

            # Apply dimension-specific scaling
            if str(dim) in self.dim_scales:
                truncated = truncated * self.dim_scales[str(dim)]

            if normalize:
                truncated = F.normalize(truncated, p=2, dim=-1)

            results[dim] = truncated

        return results

    def get_optimal_dim_for_device(
        self,
        memory_mb: Optional[float] = None,
        latency_ms: Optional[float] = None,
        num_vectors: int = 10000
    ) -> int:
        """
        Recommend optimal dimension based on device constraints.

        Args:
            memory_mb: Available memory in MB
            latency_ms: Target query latency in ms
            num_vectors: Expected number of vectors in index

        Returns:
            Recommended dimension from matryoshka_dims
        """
        # Memory estimation: dim * 4 bytes (FP32) or dim * 2 bytes (FP16)
        bytes_per_dim_fp16 = 2

        for dim in self.matryoshka_dims:  # Already sorted descending
            estimated_memory_mb = (num_vectors * dim * bytes_per_dim_fp16) / (1024 * 1024)

            if memory_mb is not None and estimated_memory_mb > memory_mb:
                continue

            # Latency roughly proportional to dimension for cosine similarity
            # Assume ~0.001ms per dimension for 10K vectors
            estimated_latency = dim * 0.001

            if latency_ms is not None and estimated_latency > latency_ms:
                continue

            return dim

        # Return smallest dimension if constraints very tight
        return min(self.matryoshka_dims)

    def save(self, path: str):
        """Save Matryoshka projection layer."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'matryoshka_dims': self.matryoshka_dims,
            'max_dim': self.max_dim,
            'use_mlp': self.use_mlp,
            'dropout': self.dropout,
        }, path)
        print(f"Matryoshka projection saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'MatryoshkaProjection':
        """Load Matryoshka projection from disk."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            input_dim=checkpoint['input_dim'],
            matryoshka_dims=checkpoint['matryoshka_dims'],
            use_mlp=checkpoint.get('use_mlp', True),
            dropout=checkpoint.get('dropout', 0.1),
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        print(f"Matryoshka projection loaded from {path}")
        return model

    def __repr__(self) -> str:
        return (
            f"MatryoshkaProjection("
            f"input_dim={self.input_dim}, "
            f"dims={self.matryoshka_dims}, "
            f"use_mlp={self.use_mlp})"
        )


class MatryoshkaLoss(nn.Module):
    """
    Multi-scale contrastive loss for training Matryoshka embeddings.

    Combines losses at multiple dimensions to ensure all truncated
    versions maintain good semantic properties.
    """

    def __init__(
        self,
        matryoshka_dims: List[int] = None,
        dim_weights: Optional[List[float]] = None,
        temperature: float = 0.07
    ):
        """
        Initialize Matryoshka loss.

        Args:
            matryoshka_dims: Dimensions to compute loss at
            dim_weights: Weight for each dimension's loss (default: equal)
            temperature: Temperature for contrastive loss
        """
        super().__init__()
        self.matryoshka_dims = matryoshka_dims or DEFAULT_MATRYOSHKA_DIMS
        self.temperature = temperature

        if dim_weights is None:
            # Default: weight larger dimensions slightly more
            dim_weights = [1.0 / (i + 1) for i in range(len(self.matryoshka_dims))]

        # Normalize weights
        total = sum(dim_weights)
        self.dim_weights = [w / total for w in dim_weights]

    def contrastive_loss(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss.

        Args:
            embeddings_a: First set of embeddings [batch_size, dim]
            embeddings_b: Second set of embeddings [batch_size, dim]

        Returns:
            Scalar loss value
        """
        # Normalize
        embeddings_a = F.normalize(embeddings_a, p=2, dim=-1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=-1)

        # Compute similarity matrix
        logits = torch.matmul(embeddings_a, embeddings_b.T) / self.temperature

        # Labels: diagonal elements are positive pairs
        batch_size = embeddings_a.size(0)
        labels = torch.arange(batch_size, device=embeddings_a.device)

        # Cross entropy loss (both directions)
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.T, labels)

        return (loss_a + loss_b) / 2

    def forward(
        self,
        multi_scale_a: dict,
        multi_scale_b: dict
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute multi-scale Matryoshka loss.

        Args:
            multi_scale_a: Dict of {dim: embeddings} for first input
            multi_scale_b: Dict of {dim: embeddings} for second input

        Returns:
            total_loss: Weighted sum of losses at all dimensions
            loss_dict: Individual losses for logging
        """
        total_loss = 0.0
        loss_dict = {}

        for dim, weight in zip(self.matryoshka_dims, self.dim_weights):
            if dim not in multi_scale_a or dim not in multi_scale_b:
                continue

            dim_loss = self.contrastive_loss(multi_scale_a[dim], multi_scale_b[dim])
            total_loss = total_loss + weight * dim_loss
            loss_dict[f'loss_{dim}d'] = dim_loss.item()

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


def get_matryoshka_dim_for_mobile_tier(tier: str) -> int:
    """
    Get recommended Matryoshka dimension for mobile device tier.

    Args:
        tier: Device tier ('high', 'mid', 'low', 'ultra_low')

    Returns:
        Recommended embedding dimension
    """
    tier_mapping = {
        'high': 512,      # Flagship phones (8GB+ RAM)
        'mid': 256,       # Mid-range phones (4-6GB RAM)
        'low': 128,       # Budget phones (2-4GB RAM)
        'ultra_low': 64,  # Very constrained devices
    }
    return tier_mapping.get(tier.lower(), 256)
