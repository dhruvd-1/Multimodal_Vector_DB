import librosa
import numpy as np
import torch
from typing import List
from .base_embedder import BaseEmbedder


class AudioEmbedder(BaseEmbedder):
    """Audio embedder using mel spectrogram features and CNN."""

    def __init__(self,
                 model_name: str = "audio-cnn",
                 device: str = 'cpu',
                 sample_rate: int = 16000,
                 n_mels: int = 128,
                 hop_length: int = 512,
                 duration: float = 10.0):
        """
        Initialize audio embedder.

        Args:
            model_name: Model identifier
            device: Device to run on
            sample_rate: Target sample rate
            n_mels: Number of mel bands
            hop_length: Hop length for STFT
            duration: Max audio duration in seconds
        """
        super().__init__(model_name, device)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.duration = duration
        self.embedding_dim = 512

    def load_model(self):
        """Load or initialize audio embedding model."""
        print(f"Loading audio embedder: {self.model_name}")

        # For now, we'll use feature-based approach
        # In production, you'd load a pretrained model like wav2vec2 or CLAP

        # Simple CNN for demonstration
        self.model = self._build_simple_cnn()
        self.model.to(self.device)
        self.model.eval()

        print("Audio embedder loaded")

    def _build_simple_cnn(self):
        """Build a simple CNN for audio embeddings."""
        import torch.nn as nn

        class SimpleCNN(nn.Module):
            def __init__(self, n_mels=128, embedding_dim=512):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((4, 4))
                self.fc = nn.Linear(128 * 4 * 4, embedding_dim)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = nn.functional.max_pool2d(x, 2)
                x = self.relu(self.conv2(x))
                x = nn.functional.max_pool2d(x, 2)
                x = self.relu(self.conv3(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        return SimpleCNN(self.n_mels, self.embedding_dim)

    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load and preprocess audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Audio waveform
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)

        # Pad or truncate to fixed length
        target_length = int(self.sample_rate * self.duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]

        return audio

    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute mel spectrogram.

        Args:
            audio: Audio waveform

        Returns:
            Mel spectrogram
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length
        )

        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)

        return mel_spec_db

    def embed(self, audio_path: str) -> np.ndarray:
        """
        Generate embedding for audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Audio embedding vector (normalized)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Load audio
        audio = self.load_audio(audio_path)

        # Compute mel spectrogram
        mel_spec = self.compute_mel_spectrogram(audio)

        # Convert to tensor
        mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)  # [1, 1, n_mels, time]
        mel_tensor = mel_tensor.to(self.device)

        # Get embedding
        with torch.no_grad():
            embedding = self.model(mel_tensor)
            # Normalize
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu().numpy()[0]

    def batch_embed(self, audio_paths: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Generate embeddings for batch of audio files.

        Args:
            audio_paths: List of audio file paths
            batch_size: Batch size for processing

        Returns:
            Matrix of audio embedding vectors
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

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
            mel_tensor = torch.FloatTensor(np.array(mel_specs)).unsqueeze(1)  # [batch, 1, n_mels, time]
            mel_tensor = mel_tensor.to(self.device)

            # Get embeddings
            with torch.no_grad():
                embeddings = self.model(mel_tensor)
                # Normalize
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)
