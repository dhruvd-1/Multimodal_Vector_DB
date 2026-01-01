"""
Projection layer for dimensionality expansion (512D -> 1024D).
"""

import torch
import torch.nn as nn
import os


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
        """
        Save projection layer weights.

        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim
        }, path)
        print(f"Projection layer saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        """
        Load projection layer from disk.

        Args:
            path: Path to load the model from
            device: Device to load the model on

        Returns:
            Loaded ProjectionLayer instance
        """
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            input_dim=checkpoint['input_dim'],
            output_dim=checkpoint['output_dim']
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        print(f"Projection layer loaded from {path}")
        return model
