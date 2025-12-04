import numpy as np
from typing import Tuple


class ProductQuantizer:
    """Product Quantization for vector compression."""

    def __init__(self, dimension: int, n_subquantizers: int = 8, n_bits: int = 8):
        """
        Initialize product quantizer.

        Args:
            dimension: Vector dimension
            n_subquantizers: Number of subquantizers
            n_bits: Bits per subquantizer
        """
        if dimension % n_subquantizers != 0:
            raise ValueError("Dimension must be divisible by n_subquantizers")

        self.dimension = dimension
        self.n_subquantizers = n_subquantizers
        self.n_bits = n_bits
        self.subdimension = dimension // n_subquantizers
        self.n_centroids = 2 ** n_bits

        self.codebooks = None  # Will store centroids for each subquantizer

    def train(self, vectors: np.ndarray, n_iterations: int = 10):
        """
        Train product quantizer using k-means.

        Args:
            vectors: Training vectors
            n_iterations: K-means iterations
        """
        from scipy.cluster.vq import kmeans2

        vectors = vectors.astype('float32')
        n_samples = len(vectors)

        print(f"Training PQ with {self.n_subquantizers} subquantizers...")

        self.codebooks = []

        for i in range(self.n_subquantizers):
            start_dim = i * self.subdimension
            end_dim = (i + 1) * self.subdimension

            # Extract subvector
            subvectors = vectors[:, start_dim:end_dim]

            # Run k-means
            centroids, _ = kmeans2(subvectors, self.n_centroids, iter=n_iterations)

            self.codebooks.append(centroids)

        print("PQ training complete")

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        Encode vectors to compact codes.

        Args:
            vectors: Vectors to encode

        Returns:
            Encoded codes (shape: [n, n_subquantizers])
        """
        if self.codebooks is None:
            raise RuntimeError("Quantizer not trained")

        vectors = vectors.astype('float32')
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        codes = np.zeros((len(vectors), self.n_subquantizers), dtype=np.uint8)

        for i in range(self.n_subquantizers):
            start_dim = i * self.subdimension
            end_dim = (i + 1) * self.subdimension

            subvectors = vectors[:, start_dim:end_dim]

            # Find nearest centroid for each subvector
            distances = np.linalg.norm(
                subvectors[:, None, :] - self.codebooks[i][None, :, :],
                axis=2
            )
            codes[:, i] = np.argmin(distances, axis=1)

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Decode compact codes back to vectors.

        Args:
            codes: Encoded codes

        Returns:
            Decoded vectors (approximate)
        """
        if self.codebooks is None:
            raise RuntimeError("Quantizer not trained")

        if codes.ndim == 1:
            codes = codes.reshape(1, -1)

        vectors = np.zeros((len(codes), self.dimension), dtype=np.float32)

        for i in range(self.n_subquantizers):
            start_dim = i * self.subdimension
            end_dim = (i + 1) * self.subdimension

            # Lookup centroids
            vectors[:, start_dim:end_dim] = self.codebooks[i][codes[:, i]]

        return vectors

    def compute_compression_ratio(self) -> float:
        """Calculate compression ratio."""
        original_size = self.dimension * 4  # float32
        compressed_size = self.n_subquantizers * 1  # uint8
        return original_size / compressed_size
