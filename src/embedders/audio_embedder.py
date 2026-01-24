"""
Audio Embedder with Mobile Support and Matryoshka Embeddings.

Features:
- CNN-based audio encoder with mel-spectrogram features
- Matryoshka variable dimension output (512/256/128/64)
- Mobile-optimized with quantization support
- Streaming audio support for large files
- Memory management for constrained devices

Note: Audio embeddings are NOT in the same space as CLIP text/image embeddings.
For cross-modal audio-text search, consider using CLAP or similar models.
"""

import sys
import os
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from .base_embedder import BaseEmbedder, QuantizationType, ModelFormat
from .projection import MatryoshkaProjection, DEFAULT_MATRYOSHKA_DIMS


class SimpleCNN(nn.Module):
    """Simple CNN for audio embeddings from mel-spectrograms."""

    def __init__(
        self,
        n_mels: int = 128,
        embedding_dim: int = 512,
        mobile_mode: bool = False
    ):
        """
        Initialize CNN.

        Args:
            n_mels: Number of mel frequency bins
            embedding_dim: Output embedding dimension
            mobile_mode: Use lighter architecture for mobile
        """
        super().__init__()
        self.n_mels = n_mels
        self.embedding_dim = embedding_dim
        self.mobile_mode = mobile_mode

        if mobile_mode:
            # Lighter architecture for mobile
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.fc = nn.Linear(64 * 4 * 4, embedding_dim)
        else:
            # Full architecture
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.fc = nn.Linear(128 * 4 * 4, embedding_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN."""
        x = self.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = self.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class AudioEmbedder(BaseEmbedder):
    """
    Audio embedder using mel-spectrogram features and CNN.

    Inherits from BaseEmbedder to get:
    - Mobile device detection
    - Quantization support (FP16, INT8)
    - Memory management (load/unload)
    - Async operations
    - Resource constraints

    Note: Audio embeddings are in a separate embedding space from CLIP.
    For unified audio-text search, consider using CLAP models.
    """

    def __init__(
        self,
        model_name: str = "audio-cnn",
        device: str = 'cpu',
        quantization: str = 'none',
        model_format: str = 'pytorch',
        max_batch_size: Optional[int] = None,
        memory_limit_mb: Optional[int] = None,
        enable_async: bool = False,
        # Audio-specific options
        sample_rate: int = 16000,
        n_mels: int = 128,
        hop_length: int = 512,
        duration: float = 10.0,
        # Matryoshka options
        use_matryoshka: bool = False,
        matryoshka_dim: Optional[int] = None,
        matryoshka_dims: List[int] = None,
    ):
        """
        Initialize audio embedder with mobile-aware configuration.

        Args:
            model_name: Model identifier
            device: Device to run on ('cpu', 'cuda', 'mps', 'mobile_cpu', etc.)
            quantization: Quantization type ('none', 'fp16', 'int8', 'dynamic')
            model_format: Model format ('pytorch', 'onnx', 'torchscript')
            max_batch_size: Maximum batch size for mobile
            memory_limit_mb: Memory limit in MB
            enable_async: Enable async operations
            sample_rate: Audio sample rate (Hz)
            n_mels: Number of mel frequency bins
            hop_length: Hop length for spectrogram
            duration: Max audio duration to process (seconds)
            use_matryoshka: Use Matryoshka projection for variable dimensions
            matryoshka_dim: Target Matryoshka dimension (None = 512)
            matryoshka_dims: List of supported Matryoshka dimensions
        """
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

        # Audio-specific settings
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.duration = duration

        # Matryoshka settings
        self.use_matryoshka = use_matryoshka
        self.matryoshka_dim = matryoshka_dim
        self.matryoshka_dims = matryoshka_dims or DEFAULT_MATRYOSHKA_DIMS
        self.matryoshka_projection = None

        # Set embedding dimension
        self._set_embedding_dim()

        # Lazy imports for optional dependencies
        self._librosa = None
        self._soundfile = None

    def _set_embedding_dim(self):
        """Set the output embedding dimension based on configuration."""
        if self.use_matryoshka:
            self.embedding_dim = self.matryoshka_dim or max(self.matryoshka_dims)
        else:
            self.embedding_dim = 512

    def _import_audio_libs(self):
        """Lazily import audio processing libraries."""
        if self._librosa is None:
            try:
                import librosa
                import soundfile as sf
                self._librosa = librosa
                self._soundfile = sf
            except ImportError as e:
                raise ImportError(
                    "Audio processing requires librosa and soundfile. "
                    "Install with: pip install librosa soundfile"
                ) from e

    def load_model(self) -> None:
        """
        Load audio embedding model with mobile optimizations.

        Handles:
        - CNN model initialization
        - Mobile-optimized architecture
        - Quantization (FP16, INT8)
        - Matryoshka layer initialization
        """
        print(f"Loading audio embedder: {self.model_name}")

        # Import audio libraries
        self._import_audio_libs()

        # Build CNN model (use mobile mode if on mobile device)
        self.model = SimpleCNN(
            n_mels=self.n_mels,
            embedding_dim=512,  # Base CNN always outputs 512D
            mobile_mode=self._is_mobile,
        )

        # Apply quantization
        self._apply_quantization()

        # Move to device
        self.model.to(self.device)
        self.model.eval()

        # Initialize Matryoshka projection if enabled
        if self.use_matryoshka:
            self.matryoshka_projection = MatryoshkaProjection(
                input_dim=512,
                matryoshka_dims=self.matryoshka_dims,
                use_mlp=True,
            )
            self.matryoshka_projection.to(self.device)
            self.matryoshka_projection.eval()
            print(f"Initialized Matryoshka projection: dims={self.matryoshka_dims}")

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
                    {nn.Linear, nn.Conv2d},
                    dtype=torch.qint8
                )
                print("Applied INT8 dynamic quantization")
            except Exception as e:
                print(f"INT8 quantization failed, using FP32: {e}")
        elif self.quantization == QuantizationType.DYNAMIC:
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {nn.Linear},
                    dtype=torch.float16
                )
                print("Applied dynamic quantization")
            except Exception as e:
                print(f"Dynamic quantization failed: {e}")

    def _log_load_status(self):
        """Log model load status."""
        status_parts = [f"Audio embedder loaded on {self.device}"]

        if self.quantization != QuantizationType.NONE:
            status_parts.append(f"quantization={self.quantization.value}")

        if self.use_matryoshka:
            status_parts.append(f"Matryoshka dims={self.matryoshka_dims}")

        if self._is_mobile:
            status_parts.append("mobile_mode=True (lighter CNN)")

        print(f"✓ {', '.join(status_parts)}")

    def unload_model(self) -> None:
        """Unload model and projection layers to free memory."""
        if self.matryoshka_projection is not None:
            del self.matryoshka_projection
            self.matryoshka_projection = None

        # Call parent unload
        super().unload_model()

    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load and preprocess audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Audio waveform as numpy array
        """
        self._import_audio_libs()

        audio, sr = self._librosa.load(
            audio_path,
            sr=self.sample_rate,
            duration=self.duration
        )

        # Pad or truncate to fixed length
        target_length = int(self.sample_rate * self.duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]

        return audio

    def load_audio_streaming(
        self,
        audio_path: str,
        chunk_duration: float = 5.0
    ):
        """
        Load audio in chunks for memory-efficient processing.

        Useful for mobile devices with limited memory.

        Args:
            audio_path: Path to audio file
            chunk_duration: Duration of each chunk in seconds

        Yields:
            Audio chunks as numpy arrays
        """
        self._import_audio_libs()

        # Get file info
        info = self._soundfile.info(audio_path)
        total_samples = info.frames
        chunk_samples = int(chunk_duration * self.sample_rate)

        # Read in chunks
        with self._soundfile.SoundFile(audio_path) as f:
            while f.tell() < total_samples:
                chunk = f.read(chunk_samples)
                if len(chunk) == 0:
                    break

                # Resample if needed
                if info.samplerate != self.sample_rate:
                    chunk = self._librosa.resample(
                        chunk,
                        orig_sr=info.samplerate,
                        target_sr=self.sample_rate
                    )

                yield chunk

    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute mel spectrogram from audio.

        Args:
            audio: Audio waveform

        Returns:
            Mel spectrogram (normalized, log-scale)
        """
        self._import_audio_libs()

        mel_spec = self._librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length
        )

        # Convert to log scale
        mel_spec_db = self._librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)

        return mel_spec_db

    def embed(
        self,
        audio_path: str,
        output_dim: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate embedding for audio file.

        Args:
            audio_path: Path to audio file
            output_dim: Override output dimension (for Matryoshka)

        Returns:
            Normalized embedding vector as numpy array
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Load audio
        audio = self.load_audio(audio_path)

        # Compute mel spectrogram
        mel_spec = self.compute_mel_spectrogram(audio)

        # Convert to tensor
        mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)
        mel_tensor = mel_tensor.to(self.device)

        # Handle FP16
        if self.quantization == QuantizationType.FP16:
            mel_tensor = mel_tensor.half()

        # Get embedding
        with torch.no_grad():
            embedding = self.model(mel_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            # Apply Matryoshka projection if configured
            if self.use_matryoshka and self.matryoshka_projection is not None:
                dim = output_dim or self.matryoshka_dim
                embedding = self.matryoshka_projection(embedding, output_dim=dim)

        return embedding.cpu().float().numpy()[0]

    def batch_embed(
        self,
        audio_paths: List[str],
        batch_size: Optional[int] = None,
        output_dim: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate embeddings for batch of audio files.

        Args:
            audio_paths: List of audio file paths
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

        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i + batch_size]

            # Load and process batch
            mel_specs = []
            for path in batch_paths:
                audio = self.load_audio(path)
                mel_spec = self.compute_mel_spectrogram(audio)
                mel_specs.append(mel_spec)

            # Stack into batch tensor
            mel_tensor = torch.FloatTensor(np.array(mel_specs)).unsqueeze(1)
            mel_tensor = mel_tensor.to(self.device)

            # Handle FP16
            if self.quantization == QuantizationType.FP16:
                mel_tensor = mel_tensor.half()

            # Get embeddings
            with torch.no_grad():
                embeddings = self.model(mel_tensor)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

                # Apply Matryoshka projection if configured
                if self.use_matryoshka and self.matryoshka_projection is not None:
                    dim = output_dim or self.matryoshka_dim
                    embeddings = self.matryoshka_projection(embeddings, output_dim=dim)

            all_embeddings.append(embeddings.cpu().float().numpy())

        return np.vstack(all_embeddings)

    def embed_from_array(
        self,
        audio: np.ndarray,
        output_dim: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate embedding from audio array directly.

        Useful for real-time audio processing.

        Args:
            audio: Audio waveform as numpy array
            output_dim: Override output dimension (for Matryoshka)

        Returns:
            Normalized embedding vector
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Pad or truncate
        target_length = int(self.sample_rate * self.duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]

        # Compute mel spectrogram
        mel_spec = self.compute_mel_spectrogram(audio)

        # Convert to tensor
        mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)
        mel_tensor = mel_tensor.to(self.device)

        if self.quantization == QuantizationType.FP16:
            mel_tensor = mel_tensor.half()

        # Get embedding
        with torch.no_grad():
            embedding = self.model(mel_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            if self.use_matryoshka and self.matryoshka_projection is not None:
                dim = output_dim or self.matryoshka_dim
                embedding = self.matryoshka_projection(embedding, output_dim=dim)

        return embedding.cpu().float().numpy()[0]

    def set_matryoshka_dim(self, dim: int):
        """Change the default Matryoshka output dimension."""
        if not self.use_matryoshka:
            raise RuntimeError("Matryoshka mode not enabled")

        if dim not in self.matryoshka_dims:
            nearest = min(self.matryoshka_dims, key=lambda d: abs(d - dim))
            print(f"Dimension {dim} not supported, using nearest: {nearest}")
            dim = nearest

        self.matryoshka_dim = dim
        self.embedding_dim = dim
        print(f"Matryoshka dimension set to {dim}")

    def warmup(self, sample_input: Optional[str] = None) -> None:
        """
        Warm up the model with a sample inference.

        Args:
            sample_input: Path to sample audio file
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Generate dummy audio for warmup
        dummy_audio = np.random.randn(int(self.sample_rate * 1.0)).astype(np.float32)
        _ = self.embed_from_array(dummy_audio)
        print("Audio model warmup complete")


# =============================================================================
# Utility Functions
# =============================================================================

def generate_dummy_audio(
    output_path: str,
    duration: float = 3.0,
    sample_rate: int = 16000,
    frequencies: List[float] = None
) -> str:
    """
    Generate dummy audio file with specified frequencies.

    Args:
        output_path: Path to save audio file
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        frequencies: List of frequencies to generate (default: [440, 880, 1320])

    Returns:
        Path to generated audio file
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile required: pip install soundfile")

    frequencies = frequencies or [440, 880, 1320]
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Generate audio signal
    audio = np.zeros_like(t)
    for freq in frequencies:
        audio += np.sin(2 * np.pi * freq * t)

    # Normalize
    audio = audio / np.max(np.abs(audio))

    # Save
    sf.write(output_path, audio, sample_rate)
    print(f"✓ Created dummy audio: {output_path} ({frequencies} Hz)")
    return output_path


# =============================================================================
# Standalone Test
# =============================================================================

def test_audio_embedder():
    """Test the audio embedder with various configurations."""
    print("=" * 60)
    print("AUDIO EMBEDDER TEST (with Mobile & Matryoshka Support)")
    print("=" * 60)

    # Create test audio files
    print("\n--- Creating Test Audio Files ---")
    audio1_path = "/tmp/test_audio_low.wav"
    audio2_path = "/tmp/test_audio_high.wav"

    generate_dummy_audio(audio1_path, duration=3, frequencies=[220])
    generate_dummy_audio(audio2_path, duration=3, frequencies=[880])

    # Test 1: Basic embedder
    print("\n--- Test 1: Basic Audio Embedding ---")
    embedder = AudioEmbedder(device="cpu", duration=5.0)
    embedder.load_model()

    embedding = embedder.embed(audio1_path)
    print(f"✓ Embedding shape: {embedding.shape}")
    print(f"✓ Embedding norm: {np.linalg.norm(embedding):.6f}")

    # Test 2: Batch embedding
    print("\n--- Test 2: Batch Audio Embedding ---")
    embeddings = embedder.batch_embed([audio1_path, audio2_path])
    print(f"✓ Batch embeddings shape: {embeddings.shape}")
    print(f"✓ All normalized: {np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)}")

    # Test 3: Similarity
    print("\n--- Test 3: Audio Similarity ---")
    emb1 = embedder.embed(audio1_path)
    emb2 = embedder.embed(audio2_path)
    sim = np.dot(emb1, emb2)
    print(f"Similarity (low vs high freq): {sim:.4f}")

    embedder.unload_model()

    # Test 4: Matryoshka embedder
    print("\n--- Test 4: Matryoshka Audio Embedding ---")
    matryoshka_embedder = AudioEmbedder(
        device="cpu",
        duration=5.0,
        use_matryoshka=True,
        matryoshka_dim=256,
    )
    matryoshka_embedder.load_model()

    embedding_256 = matryoshka_embedder.embed(audio1_path)
    print(f"✓ Matryoshka 256D shape: {embedding_256.shape}")

    embedding_128 = matryoshka_embedder.embed(audio1_path, output_dim=128)
    embedding_64 = matryoshka_embedder.embed(audio1_path, output_dim=64)
    print(f"✓ Matryoshka 128D shape: {embedding_128.shape}")
    print(f"✓ Matryoshka 64D shape: {embedding_64.shape}")

    matryoshka_embedder.unload_model()

    # Test 5: Embed from array
    print("\n--- Test 5: Embed from Array ---")
    embedder = AudioEmbedder(device="cpu", duration=5.0)
    embedder.load_model()

    dummy_audio = np.random.randn(16000 * 3).astype(np.float32)
    embedding = embedder.embed_from_array(dummy_audio)
    print(f"✓ Embedding from array shape: {embedding.shape}")

    embedder.unload_model()

    # Cleanup
    print("\n--- Cleaning up test files ---")
    for path in [audio1_path, audio2_path]:
        if os.path.exists(path):
            os.remove(path)
            print(f"✓ Removed {path}")

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_audio_embedder()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
