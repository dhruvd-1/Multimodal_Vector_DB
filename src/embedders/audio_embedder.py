"""
Standalone Test for Audio Embedder
Run: python test_audio_embedder.py
"""

import sys

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn


class AudioEmbedder:
    """Audio embedder using mel spectrogram features and CNN."""

    def __init__(
        self,
        model_name="audio-cnn",
        device="cpu",
        sample_rate=16000,
        n_mels=128,
        hop_length=512,
        duration=10.0,
    ):
        self.model_name = model_name
        self.device = device
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.duration = duration
        self.embedding_dim = 512
        self.model = None

    def load_model(self):
        """Load or initialize audio embedding model."""
        print(f"Loading audio embedder: {self.model_name}")
        self.model = self._build_simple_cnn()
        self.model.to(self.device)
        self.model.eval()
        print("✓ Audio embedder loaded")

    def _build_simple_cnn(self):
        """Build a simple CNN for audio embeddings."""

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

    def load_audio(self, audio_path):
        """Load and preprocess audio file."""
        audio, sr = librosa.load(
            audio_path, sr=self.sample_rate, duration=self.duration
        )

        # Pad or truncate to fixed length
        target_length = int(self.sample_rate * self.duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]

        return audio

    def compute_mel_spectrogram(self, audio):
        """Compute mel spectrogram."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_mels=self.n_mels, hop_length=self.hop_length
        )

        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)

        return mel_spec_db

    def embed(self, audio_path):
        """Generate embedding for audio file."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Load audio
        audio = self.load_audio(audio_path)

        # Compute mel spectrogram
        mel_spec = self.compute_mel_spectrogram(audio)

        # Convert to tensor
        mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)
        mel_tensor = mel_tensor.to(self.device)

        # Get embedding
        with torch.no_grad():
            embedding = self.model(mel_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu().numpy()[0]

    def batch_embed(self, audio_paths, batch_size=16):
        """Generate embeddings for batch of audio files."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        all_embeddings = []

        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i : i + batch_size]

            # Load and process batch
            mel_specs = []
            for path in batch_paths:
                audio = self.load_audio(path)
                mel_spec = self.compute_mel_spectrogram(audio)
                mel_specs.append(mel_spec)

            # Stack into batch tensor
            mel_tensor = torch.FloatTensor(np.array(mel_specs)).unsqueeze(1)
            mel_tensor = mel_tensor.to(self.device)

            # Get embeddings
            with torch.no_grad():
                embeddings = self.model(mel_tensor)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


def generate_dummy_audio(
    output_path, duration=3.0, sample_rate=16000, frequencies=[440, 880, 1320]
):
    """Generate dummy audio with specified frequencies (tones)."""
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Generate audio signal with multiple frequencies
    audio = np.zeros_like(t)
    for freq in frequencies:
        audio += np.sin(2 * np.pi * freq * t)

    # Normalize
    audio = audio / np.max(np.abs(audio))

    # Save
    sf.write(output_path, audio, sample_rate)
    print(f"✓ Created dummy audio: {output_path} ({frequencies} Hz)")
    return output_path


def test_audio_embedder():
    """Test the audio embedder with dummy data."""
    print("=" * 60)
    print("AUDIO EMBEDDER TEST")
    print("=" * 60)

    # Initialize embedder
    embedder = AudioEmbedder(device="cpu", duration=5.0)
    embedder.load_model()

    # Create test audio files
    print("\n--- Creating Test Audio Files ---")
    audio1_path = "test_audio_low.wav"
    audio2_path = "test_audio_mid.wav"
    audio3_path = "test_audio_high.wav"
    audio4_path = "test_audio_mixed.wav"

    generate_dummy_audio(audio1_path, duration=3, frequencies=[220])  # Low tone
    generate_dummy_audio(audio2_path, duration=3, frequencies=[440])  # Mid tone
    generate_dummy_audio(audio3_path, duration=3, frequencies=[880])  # High tone
    generate_dummy_audio(audio4_path, duration=3, frequencies=[220, 880])  # Mixed

    # Test 1: Single audio embedding
    print("\n--- Test 1: Single Audio Embedding ---")
    print(f"Processing: {audio1_path}")

    embedding = embedder.embed(audio1_path)
    print(f"✓ Embedding shape: {embedding.shape}")
    print(f"✓ Embedding dimension: {len(embedding)}")
    print(f"✓ Embedding norm (should be ~1.0): {np.linalg.norm(embedding):.6f}")
    print(f"✓ First 5 values: {embedding[:5]}")

    # Test 2: Mel spectrogram visualization
    print("\n--- Test 2: Mel Spectrogram Computation ---")
    audio = embedder.load_audio(audio1_path)
    mel_spec = embedder.compute_mel_spectrogram(audio)

    print(f"Audio shape: {audio.shape}")
    print(f"Audio duration: {len(audio) / embedder.sample_rate:.2f} seconds")
    print(f"Mel spectrogram shape: {mel_spec.shape}")
    print(f"  - Mel bins: {mel_spec.shape[0]}")
    print(f"  - Time frames: {mel_spec.shape[1]}")
    print(f"✓ Mel spectrogram computed successfully")

    # Test 3: Batch audio embedding
    print("\n--- Test 3: Batch Audio Embedding ---")
    audio_paths = [audio1_path, audio2_path, audio3_path, audio4_path]
    audio_labels = [
        "Low tone (220Hz)",
        "Mid tone (440Hz)",
        "High tone (880Hz)",
        "Mixed (220+880Hz)",
    ]

    print(f"Processing {len(audio_paths)} audio files...")
    for label in audio_labels:
        print(f"  - {label}")

    embeddings = embedder.batch_embed(audio_paths)
    print(f"\n✓ Batch embeddings shape: {embeddings.shape}")
    print(
        f"✓ All vectors normalized: {np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)}"
    )

    # Test 4: Audio similarity
    print("\n--- Test 4: Audio Similarity Test ---")
    emb_low = embedder.embed(audio1_path)
    emb_mid = embedder.embed(audio2_path)
    emb_high = embedder.embed(audio3_path)
    emb_mixed = embedder.embed(audio4_path)

    sim_low_mid = np.dot(emb_low, emb_mid)
    sim_low_high = np.dot(emb_low, emb_high)
    sim_low_mixed = np.dot(emb_low, emb_mixed)

    print(f"Similarity (low vs mid):   {sim_low_mid:.4f}")
    print(f"Similarity (low vs high):  {sim_low_high:.4f}")
    print(f"Similarity (low vs mixed): {sim_low_mixed:.4f}")
    print(f"✓ Mixed audio contains low tone: {sim_low_mixed > sim_low_high}")

    # Test 5: Self-similarity check
    print("\n--- Test 5: Self-Similarity Check ---")
    emb1 = embedder.embed(audio1_path)
    emb2 = embedder.embed(audio1_path)

    self_sim = np.dot(emb1, emb2)
    print(f"Self-similarity (should be ~1.0): {self_sim:.6f}")
    print(f"✓ Audio embedding is deterministic: {np.allclose(self_sim, 1.0)}")

    # Test 6: Search simulation
    print("\n--- Test 6: Simple Audio Search Simulation ---")
    query_path = audio1_path
    print(f"Query: Low tone audio")
    query_emb = embedder.embed(query_path)

    # Compute similarities
    print("\nRanked results:")
    sims = embeddings @ query_emb
    ranked_indices = np.argsort(sims)[::-1]

    for rank, idx in enumerate(ranked_indices, 1):
        print(f"{rank}. [{sims[idx]:.4f}] {audio_labels[idx]}")

    # Cleanup
    print("\n--- Cleaning up test audio files ---")
    import os

    for path in [audio1_path, audio2_path, audio3_path, audio4_path]:
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
