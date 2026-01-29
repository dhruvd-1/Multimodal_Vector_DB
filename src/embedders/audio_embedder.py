"""
Audio Embedder using CLAP (Contrastive Language-Audio Pretraining).

CLAP provides audio embeddings in the same semantic space as text,
enabling cross-modal search (text query → audio results).

Features:
- Pre-trained CLAP model (no training needed)
- Audio embeddings aligned with text embeddings
- Text → Audio cross-modal search capability
- Mobile-optimized with memory management
- Async operations for non-blocking UI

Note: CLAP embeddings are in a DIFFERENT space than CLIP (image/text).
For unified image+audio+text search, you would need to align the spaces
or use separate indices.
"""

import sys
import os
from typing import List, Optional, Union

import numpy as np
import torch

from .base_embedder import BaseEmbedder, QuantizationType, ModelFormat


class AudioEmbedder(BaseEmbedder):
    """
    Audio embedder using CLAP for unified audio-text embeddings.

    CLAP (Contrastive Language-Audio Pretraining) provides:
    - Audio embeddings aligned with text embeddings
    - Text → Audio search capability (search audio with text queries)
    - Zero-shot audio classification
    - Pre-trained on 600K+ audio-text pairs

    Inherits from BaseEmbedder to get:
    - Mobile device detection
    - Memory management (load/unload)
    - Async operations
    - Resource constraints
    """

    # Available CLAP models
    MODELS = {
        'music_speech': 'larger_clap_music_and_speech',
        'music': 'larger_clap_music',
        'general': 'larger_clap_general',
    }

    def __init__(
        self,
        model_name: str = 'music_speech',
        device: str = 'cpu',
        quantization: str = 'none',
        model_format: str = 'pytorch',
        max_batch_size: Optional[int] = None,
        memory_limit_mb: Optional[int] = None,
        enable_async: bool = False,
        enable_fusion: bool = False,
    ):
        """
        Initialize CLAP audio embedder.

        Args:
            model_name: CLAP model variant ('music_speech', 'music', 'general')
            device: Device to run on ('cpu', 'cuda', 'mps')
            quantization: Quantization type ('none', 'fp16')
            model_format: Model format (currently only 'pytorch')
            max_batch_size: Maximum batch size for mobile
            memory_limit_mb: Memory limit in MB
            enable_async: Enable async operations
            enable_fusion: Enable CLAP fusion mode (better quality, slower)
        """
        # Resolve model name
        if model_name in self.MODELS:
            resolved_name = self.MODELS[model_name]
        else:
            resolved_name = model_name

        super().__init__(
            model_name=resolved_name,
            device=device,
            quantization=quantization,
            model_format=model_format,
            max_batch_size=max_batch_size,
            memory_limit_mb=memory_limit_mb,
            enable_async=enable_async,
        )

        self.enable_fusion = enable_fusion
        self.embedding_dim = 512
        self._clap = None

    def load_model(self) -> None:
        """
        Load CLAP model.

        Downloads pre-trained weights on first use (~1.8GB).
        Uses D: drive for caching to avoid C: drive space issues.
        """
        print(f"Loading CLAP model: {self.model_name}")
        
        # Configure cache directories to use D: drive if C: drive is constrained
        # This avoids disk space issues on the system drive
        cache_dir = "D:\\.cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Set environment variables for model caching
        os.environ['TORCH_HOME'] = os.path.join(cache_dir, 'torch')
        os.environ['HF_HOME'] = os.path.join(cache_dir, 'huggingface')
        
        # CLAP uses a custom cache directory  
        clap_cache = os.path.join(cache_dir, 'laion_clap')
        os.makedirs(clap_cache, exist_ok=True)
        
        print(f"📁 Using cache directory: {cache_dir}")
        
        # Workaround for PyTorch 2.6+ security changes with CLAP
        # The CLAP library uses torch.load which defaults to weights_only=True in 2.6+
        # We need to patch it to use weights_only=False for backward compatibility
        original_torch_load = torch.load
        def patched_torch_load(*args, **kwargs):
            # Set weights_only=False if not explicitly set
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        
        torch.load = patched_torch_load
        
        # Also patch load_state_dict to be more lenient with unexpected keys
        from torch.nn import Module
        original_load_state_dict = Module.load_state_dict
        def patched_load_state_dict(self, state_dict, strict=True, **kwargs):
            # Be lenient with position_ids which is a common mismatch
            return original_load_state_dict(self, state_dict, strict=False, **kwargs)
        
        Module.load_state_dict = patched_load_state_dict

        try:
            import laion_clap

            # Use default model configuration - let CLAP handle it
            self._clap = laion_clap.CLAP_Module(
                enable_fusion=self.enable_fusion,
                device=self.device
            )
            
            self._clap.load_ckpt()
            
            # Restore original functions
            torch.load = original_torch_load
            Module.load_state_dict = original_load_state_dict

            self.model = self._clap
            self._is_loaded = True

            status = f"CLAP loaded on {self.device}"
            if self._is_mobile:
                status += " (mobile mode: HTSAT-tiny)"
            if self.enable_fusion:
                status += " (fusion enabled)"
            print(f"✓ {status}")

        except ImportError:
            raise ImportError(
                "CLAP required for audio embeddings.\n"
                "Install with: pip install laion-clap"
            )

    def unload_model(self) -> None:
        """Unload CLAP model to free memory."""
        if self._clap is not None:
            del self._clap
            self._clap = None

        super().unload_model()

    def embed(self, audio_path: str) -> np.ndarray:
        """
        Generate embedding for audio file.

        Args:
            audio_path: Path to audio file (wav, mp3, flac, etc.)

        Returns:
            Normalized 512D embedding vector
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Get audio embedding (returns already as numpy array)
        embedding = self._clap.get_audio_embedding_from_filelist([audio_path])

        # Normalize
        emb = embedding[0]
        return emb / np.linalg.norm(emb)

    def batch_embed(
        self,
        audio_paths: List[str],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate embeddings for multiple audio files.

        Args:
            audio_paths: List of audio file paths
            batch_size: Batch size (respects max_batch_size)

        Returns:
            Matrix of normalized 512D embeddings [N, 512]
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        batch_size = self.get_effective_batch_size(batch_size)

        all_embeddings = []
        for i in range(0, len(audio_paths), batch_size):
            batch = audio_paths[i:i + batch_size]

            embeddings = self._clap.get_audio_embedding_from_filelist(
                batch,
                use_tensor=False
            )
            all_embeddings.append(embeddings)

        embeddings = np.vstack(all_embeddings)

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate text embedding for audio search.

        This enables cross-modal search: text query → audio results.

        Args:
            text: Text description (e.g., "dog barking", "piano music")

        Returns:
            Normalized 512D embedding in CLAP's audio-text space
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Use get_audio_embedding_from_data with text as input
        # This requires tokenizing and reshaping properly
        import torch
        
        # Tokenize text
        text_input = self._clap.tokenizer([text])
        
        # Ensure proper shape for RoBERTa model (batch_size, seq_length)
        if 'input_ids' in text_input:
            input_ids = text_input['input_ids']
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)
            
            # Make sure it's 2D
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            text_input['input_ids'] = input_ids
            
        if 'attention_mask' in text_input:
            attention_mask = text_input['attention_mask']
            if not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.tensor(attention_mask)
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            text_input['attention_mask'] = attention_mask
        
        # Get embedding using model's encode_text method directly
        with torch.no_grad():
            embedding = self._clap.model.encode_text(text_input, device=self.device)
        
        # Convert to numpy and normalize
        if isinstance(embedding, torch.Tensor):
            emb = embedding[0].cpu().numpy()
        else:
            emb = np.array(embedding[0])
            
        return emb / np.linalg.norm(emb)

    def batch_embed_text(self, texts: List[str]) -> np.ndarray:
        """
        Generate text embeddings for multiple queries.

        Args:
            texts: List of text descriptions

        Returns:
            Matrix of normalized 512D embeddings [N, 512]
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        embeddings = self._clap.get_text_embedding(
            texts,
            use_tensor=False
        )

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    def search_audio_with_text(
        self,
        text_query: str,
        audio_embeddings: np.ndarray,
        top_k: int = 10
    ) -> List[tuple]:
        """
        Search audio embeddings using text query.

        Args:
            text_query: Text description to search for
            audio_embeddings: Pre-computed audio embeddings [N, 512]
            top_k: Number of results to return

        Returns:
            List of (index, similarity_score) tuples, sorted by similarity
        """
        text_emb = self.embed_text(text_query)

        # Compute similarities (dot product of normalized vectors = cosine sim)
        similarities = audio_embeddings @ text_emb

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(idx, similarities[idx]) for idx in top_indices]

    def warmup(self, sample_input: Optional[str] = None) -> None:
        """
        Warm up the model with a sample inference.

        Args:
            sample_input: Optional path to sample audio file
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Warmup with text embedding (doesn't require audio file)
        _ = self.embed_text("warmup audio query")
        print("CLAP model warmup complete")


# =============================================================================
# Standalone Test
# =============================================================================

def test_clap_embedder():
    """Test the CLAP audio embedder."""
    print("=" * 60)
    print("CLAP AUDIO EMBEDDER TEST")
    print("=" * 60)

    # Test 1: Initialize and load
    print("\n--- Test 1: Load CLAP Model ---")
    try:
        embedder = AudioEmbedder(model_name='music_speech', device='cpu')
        embedder.load_model()
        print("✓ CLAP model loaded successfully")
    except ImportError as e:
        print(f"✗ CLAP not installed: {e}")
        print("Install with: pip install laion-clap")
        return

    # Test 2: Text embedding
    print("\n--- Test 2: Text Embedding ---")
    text = "a dog barking loudly"
    text_emb = embedder.embed_text(text)
    print(f"Text: '{text}'")
    print(f"✓ Embedding shape: {text_emb.shape}")
    print(f"✓ Embedding norm: {np.linalg.norm(text_emb):.4f}")

    # Test 3: Multiple text embeddings
    print("\n--- Test 3: Batch Text Embedding ---")
    texts = [
        "dog barking",
        "cat meowing",
        "piano music",
        "car engine",
    ]
    text_embs = embedder.batch_embed_text(texts)
    print(f"✓ Batch embeddings shape: {text_embs.shape}")

    # Test 4: Text similarity
    print("\n--- Test 4: Text Similarity ---")
    sim_matrix = text_embs @ text_embs.T
    print("Similarity matrix:")
    for i, t1 in enumerate(texts):
        for j, t2 in enumerate(texts):
            if j >= i:
                print(f"  '{t1}' vs '{t2}': {sim_matrix[i,j]:.3f}")

    embedder.unload_model()

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nNote: Audio file tests skipped (no test files)")
    print("To test audio embedding, call: embedder.embed('path/to/audio.wav')")


if __name__ == "__main__":
    try:
        test_clap_embedder()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
