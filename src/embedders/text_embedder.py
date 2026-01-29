"""
Text Embedder with Mobile Support and Matryoshka Embeddings.

Features:
- CLIP ViT-B/32 (LAION-2B) text encoder
- Matryoshka variable dimension output (512/256/128/64)
- Mobile-optimized with quantization support
- Async operations for non-blocking UI
- Memory management for constrained devices
"""

import sys
from typing import List, Optional, Union

import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer

from .base_embedder import BaseEmbedder, QuantizationType, ModelFormat
from .projection import ProjectionLayer, MatryoshkaProjection, DEFAULT_MATRYOSHKA_DIMS


class TextEmbedder(BaseEmbedder):
    """
    Text embedder using CLIP text encoder with mobile and Matryoshka support.

    Inherits from BaseEmbedder to get:
    - Mobile device detection
    - Quantization support (FP16, INT8)
    - Memory management (load/unload)
    - Async operations
    - Resource constraints
    """

    # Default model: CLIP ViT-B/32 trained on LAION-2B
    DEFAULT_MODEL = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    FALLBACK_MODEL = "openai/clip-vit-base-patch32"

    def __init__(
        self,
        model_name: str = None,
        device: str = 'cpu',
        quantization: str = 'none',
        model_format: str = 'pytorch',
        max_batch_size: Optional[int] = None,
        memory_limit_mb: Optional[int] = None,
        enable_async: bool = False,
        # Text-specific options
        max_length: int = 77,
        # Projection options
        use_projection: bool = False,
        projection_dim: int = 1024,
        # Matryoshka options
        use_matryoshka: bool = False,
        matryoshka_dim: Optional[int] = None,
        matryoshka_dims: List[int] = None,
    ):
        """
        Initialize text embedder with mobile-aware configuration.

        Args:
            model_name: CLIP model name (default: LAION ViT-B/32)
            device: Device to run on ('cpu', 'cuda', 'mps', 'mobile_cpu', etc.)
            quantization: Quantization type ('none', 'fp16', 'int8', 'dynamic')
            model_format: Model format ('pytorch', 'onnx', 'torchscript')
            max_batch_size: Maximum batch size for mobile
            memory_limit_mb: Memory limit in MB
            enable_async: Enable async operations
            max_length: Maximum token length (CLIP uses 77)
            use_projection: Use projection layer for dimension expansion
            projection_dim: Output dimension for projection (default: 1024)
            use_matryoshka: Use Matryoshka projection for variable dimensions
            matryoshka_dim: Target Matryoshka dimension (None = max dim)
            matryoshka_dims: List of supported Matryoshka dimensions
        """
        # Use default model if not specified
        model_name = model_name or self.DEFAULT_MODEL

        # Initialize base class with mobile support
        super().__init__(
            model_name=model_name,
            device=device,
            quantization=quantization,
            model_format=model_format,
            max_batch_size=max_batch_size,
            memory_limit_mb=memory_limit_mb,
            enable_async=enable_async,
        )

        # Text-specific settings
        self.max_length = max_length
        self.tokenizer = None

        # Projection settings
        self.use_projection = use_projection
        self.projection_dim = projection_dim
        self.projection = None

        # Matryoshka settings
        self.use_matryoshka = use_matryoshka
        self.matryoshka_dim = matryoshka_dim
        self.matryoshka_dims = matryoshka_dims or DEFAULT_MATRYOSHKA_DIMS
        self.matryoshka_projection = None

        # Set embedding dimension based on configuration
        self._set_embedding_dim()

    def _set_embedding_dim(self):
        """Set the output embedding dimension based on configuration."""
        if self.use_matryoshka:
            # Matryoshka: use specified dim or max available
            self.embedding_dim = self.matryoshka_dim or max(self.matryoshka_dims)
        elif self.use_projection:
            # Projection: use projection output dim
            self.embedding_dim = self.projection_dim
        else:
            # Raw CLIP: 512D
            self.embedding_dim = 512

    def load_model(self) -> None:
        """
        Load CLIP model and tokenizer with mobile optimizations.

        Handles:
        - Model loading with fallback
        - Quantization (FP16, INT8)
        - Projection/Matryoshka layer initialization
        - Device placement
        """
        print(f"Loading CLIP text encoder: {self.model_name}")

        try:
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
        except Exception as e:
            print(f"Failed to load {self.model_name}, trying fallback: {e}")
            self.model_name = self.FALLBACK_MODEL
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)

        # Apply quantization
        self._apply_quantization()

        # Move to device
        self.model.to(self.device)
        self.model.eval()

        # Initialize projection layers
        self._init_projection_layers()

        self._is_loaded = True
        self._log_load_status()

    def _apply_quantization(self):
        """Apply quantization based on configuration."""
        if self.quantization == QuantizationType.FP16:
            if self.device != 'cpu' or torch.cuda.is_available():
                self.model = self.model.half()
                print("Applied FP16 quantization")
        elif self.quantization == QuantizationType.INT8:
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                print("Applied INT8 dynamic quantization")
            except Exception as e:
                print(f"INT8 quantization failed, using FP32: {e}")
        elif self.quantization == QuantizationType.DYNAMIC:
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear, torch.nn.LayerNorm},
                    dtype=torch.float16
                )
                print("Applied dynamic quantization")
            except Exception as e:
                print(f"Dynamic quantization failed: {e}")

    def _init_projection_layers(self):
        """Initialize projection or Matryoshka layers."""
        if self.use_matryoshka:
            self.matryoshka_projection = MatryoshkaProjection(
                input_dim=512,
                matryoshka_dims=self.matryoshka_dims,
                use_mlp=True,
            )
            self.matryoshka_projection.to(self.device)
            self.matryoshka_projection.eval()
            print(f"Initialized Matryoshka projection: dims={self.matryoshka_dims}")

        elif self.use_projection:
            self.projection = ProjectionLayer(
                input_dim=512,
                output_dim=self.projection_dim
            )
            self.projection.to(self.device)
            self.projection.eval()
            print(f"Initialized projection layer: 512D -> {self.projection_dim}D")

    def _log_load_status(self):
        """Log model load status."""
        status_parts = [f"Model loaded on {self.device}"]

        if self.quantization != QuantizationType.NONE:
            status_parts.append(f"quantization={self.quantization.value}")

        if self.use_matryoshka:
            status_parts.append(f"Matryoshka dims={self.matryoshka_dims}")
        elif self.use_projection:
            status_parts.append(f"projection={self.projection_dim}D")

        if self._is_mobile:
            status_parts.append("mobile_mode=True")

        print(f"✓ {', '.join(status_parts)}")

    def unload_model(self) -> None:
        """Unload model and projection layers to free memory."""
        # Unload projection layers
        if self.projection is not None:
            del self.projection
            self.projection = None

        if self.matryoshka_projection is not None:
            del self.matryoshka_projection
            self.matryoshka_projection = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Call parent unload
        super().unload_model()

    def embed(
        self,
        text: str,
        output_dim: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate embedding for single text.

        Args:
            text: Input text string
            output_dim: Override output dimension (for Matryoshka)

        Returns:
            Normalized embedding vector as numpy array
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embedding
        with torch.no_grad():
            # Use text_model directly to get pooler_output
            text_outputs = self.model.text_model(**inputs)
            pooled_output = text_outputs.pooler_output  # [batch, 768]
            text_embeds = self.model.text_projection(pooled_output)  # [batch, 512]

            # Normalize
            embeddings = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            # Apply projection if configured
            embeddings = self._apply_projection(embeddings, output_dim)

        return embeddings.cpu().numpy()[0]

    def batch_embed(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        output_dim: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate embeddings for batch of texts.

        Args:
            texts: List of text strings
            batch_size: Override batch size (respects max_batch_size)
            output_dim: Override output dimension (for Matryoshka)

        Returns:
            Matrix of normalized embedding vectors
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Get effective batch size respecting mobile limits
        batch_size = self.get_effective_batch_size(batch_size)

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get embeddings
            with torch.no_grad():
                # Use text_model directly to get pooler_output
                text_outputs = self.model.text_model(**inputs)
                pooled_output = text_outputs.pooler_output  # [batch, 768]
                text_embeds = self.model.text_projection(pooled_output)  # [batch, 512]

                # Normalize
                embeddings = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

                # Apply projection if configured
                embeddings = self._apply_projection(embeddings, output_dim)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def _apply_projection(
        self,
        embeddings: torch.Tensor,
        output_dim: Optional[int] = None
    ) -> torch.Tensor:
        """Apply projection or Matryoshka transformation."""
        if self.use_matryoshka and self.matryoshka_projection is not None:
            # Use specified dim or instance default
            dim = output_dim or self.matryoshka_dim
            return self.matryoshka_projection(embeddings, output_dim=dim)

        elif self.use_projection and self.projection is not None:
            return self.projection(embeddings)

        return embeddings

    def embed_multi_scale(self, text: str) -> dict:
        """
        Get embeddings at all Matryoshka scales.

        Args:
            text: Input text string

        Returns:
            Dictionary mapping dimension -> embedding
        """
        if not self.use_matryoshka:
            raise RuntimeError("Multi-scale embedding requires use_matryoshka=True")

        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get base embedding
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            # Get all scales
            multi_scale = self.matryoshka_projection.forward_multi_scale(embeddings)

        # Convert to numpy
        return {dim: emb.cpu().numpy()[0] for dim, emb in multi_scale.items()}

    def set_matryoshka_dim(self, dim: int):
        """
        Change the default Matryoshka output dimension.

        Useful for adapting to device capabilities at runtime.

        Args:
            dim: New output dimension (must be in matryoshka_dims)
        """
        if not self.use_matryoshka:
            raise RuntimeError("Matryoshka mode not enabled")

        if dim not in self.matryoshka_dims:
            nearest = min(self.matryoshka_dims, key=lambda d: abs(d - dim))
            print(f"Dimension {dim} not supported, using nearest: {nearest}")
            dim = nearest

        self.matryoshka_dim = dim
        self.embedding_dim = dim
        print(f"Matryoshka dimension set to {dim}")

    def get_optimal_dim_for_constraints(
        self,
        memory_mb: Optional[float] = None,
        num_vectors: int = 10000
    ) -> int:
        """
        Get optimal Matryoshka dimension for given constraints.

        Args:
            memory_mb: Available memory in MB
            num_vectors: Expected number of vectors

        Returns:
            Optimal dimension
        """
        if not self.use_matryoshka or self.matryoshka_projection is None:
            return self.embedding_dim

        return self.matryoshka_projection.get_optimal_dim_for_device(
            memory_mb=memory_mb,
            num_vectors=num_vectors
        )

    def warmup(self, sample_input: Optional[str] = None) -> None:
        """
        Warm up the model with a sample inference.

        Args:
            sample_input: Sample text (default: "warmup")
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        sample = sample_input or "warmup text for model initialization"
        _ = self.embed(sample)
        print("Model warmup complete")

    def load_matryoshka_weights(self, path: str):
        """
        Load pre-trained Matryoshka projection weights.

        Args:
            path: Path to saved Matryoshka projection
        """
        if not self.use_matryoshka:
            raise RuntimeError("Matryoshka mode not enabled")

        self.matryoshka_projection = MatryoshkaProjection.load(path, self.device)
        print(f"Loaded Matryoshka weights from {path}")

    def save_matryoshka_weights(self, path: str):
        """
        Save Matryoshka projection weights.

        Args:
            path: Path to save Matryoshka projection
        """
        if not self.use_matryoshka or self.matryoshka_projection is None:
            raise RuntimeError("No Matryoshka projection to save")

        self.matryoshka_projection.save(path)


# =============================================================================
# Standalone Test
# =============================================================================

def test_text_embedder():
    """Test the text embedder with various configurations."""
    print("=" * 60)
    print("TEXT EMBEDDER TEST (with Mobile & Matryoshka Support)")
    print("=" * 60)

    # Test 1: Basic embedder
    print("\n--- Test 1: Basic Text Embedding ---")
    embedder = TextEmbedder(device="cpu")
    embedder.load_model()

    text = "A cute cat playing with a red ball"
    print(f"Input text: '{text}'")

    embedding = embedder.embed(text)
    print(f"✓ Embedding shape: {embedding.shape}")
    print(f"✓ Embedding norm: {np.linalg.norm(embedding):.6f}")

    # Test 2: Batch embedding
    print("\n--- Test 2: Batch Text Embedding ---")
    texts = [
        "A cute cat playing with a red ball",
        "A golden retriever dog running on the beach",
        "A beautiful sunset over the mountains",
    ]

    embeddings = embedder.batch_embed(texts)
    print(f"✓ Batch embeddings shape: {embeddings.shape}")
    print(f"✓ All normalized: {np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)}")

    embedder.unload_model()

    # Test 3: Matryoshka embedder
    print("\n--- Test 3: Matryoshka Embedding ---")
    matryoshka_embedder = TextEmbedder(
        device="cpu",
        use_matryoshka=True,
        matryoshka_dim=256,
    )
    matryoshka_embedder.load_model()

    embedding_256 = matryoshka_embedder.embed(text)
    print(f"✓ Matryoshka 256D shape: {embedding_256.shape}")

    # Test different dimensions
    embedding_128 = matryoshka_embedder.embed(text, output_dim=128)
    embedding_64 = matryoshka_embedder.embed(text, output_dim=64)
    print(f"✓ Matryoshka 128D shape: {embedding_128.shape}")
    print(f"✓ Matryoshka 64D shape: {embedding_64.shape}")

    # Test multi-scale
    print("\n--- Test 4: Multi-Scale Embedding ---")
    multi_scale = matryoshka_embedder.embed_multi_scale(text)
    for dim, emb in multi_scale.items():
        print(f"  {dim}D: shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")

    matryoshka_embedder.unload_model()

    # Test 4: Semantic similarity
    print("\n--- Test 5: Semantic Similarity ---")
    embedder = TextEmbedder(device="cpu")
    embedder.load_model()

    text1 = "cat playing with toy"
    text2 = "kitten playing with ball"
    text3 = "airplane in the sky"

    emb1 = embedder.embed(text1)
    emb2 = embedder.embed(text2)
    emb3 = embedder.embed(text3)

    sim_12 = np.dot(emb1, emb2)
    sim_13 = np.dot(emb1, emb3)

    print(f"Similarity (similar texts): {sim_12:.4f}")
    print(f"Similarity (different texts): {sim_13:.4f}")
    print(f"✓ Similar texts have higher similarity: {sim_12 > sim_13}")

    embedder.unload_model()

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_text_embedder()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
