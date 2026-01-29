
import sys
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

from .base_embedder import BaseEmbedder, QuantizationType, ModelFormat
from .image_embedder import ImageEmbedder


class VideoEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str = None,
        device: str = 'cpu',
        quantization: str = 'fp16',
        model_format: str = 'pytorch',
        max_batch_size: Optional[int] = None,
        memory_limit_mb: Optional[int] = None,
        enable_async: bool = False,
        # Video-specific options
        sample_fps: float = 1.0,
        max_frames: int = 100,
        pooling: str = 'mean',
        use_projection: bool = False,
        projection_dim: int = 1024,
    ):
        
        model_name = model_name or "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

        super().__init__(
            model_name=model_name,
            device=device,
            quantization=quantization,
            model_format=model_format,
            max_batch_size=max_batch_size,
            memory_limit_mb=memory_limit_mb,
            enable_async=enable_async,
        )

        # Video-specific settings
        self.sample_fps = sample_fps
        self.max_frames = max_frames
        self.pooling = pooling
        self.use_projection = use_projection
        self.projection_dim = projection_dim
        self.image_embedder = None
        self.embedding_dim = projection_dim if use_projection else 512

    def load_model(self) -> None:
        print(f"Loading video embedder (CLIP: {self.model_name})")
        print(f"  Sample FPS: {self.sample_fps}")
        print(f"  Max frames: {self.max_frames}")
        print(f"  Pooling: {self.pooling}")

        self.image_embedder = ImageEmbedder(
            model_name=self.model_name,
            device=self.device,
            quantization=self.quantization.value,
            max_batch_size=self.max_batch_size,
            memory_limit_mb=self.memory_limit_mb,
            use_projection=self.use_projection,
            projection_dim=self.projection_dim,
        )
        self.image_embedder.load_model()

        self.model = self.image_embedder.model
        self._is_loaded = True


    def unload_model(self) -> None:
        if self.image_embedder is not None:
            self.image_embedder.unload_model()
            del self.image_embedder
            self.image_embedder = None

        super().unload_model()

    def extract_frames(self, video_path: str) -> List[np.ndarray]:
       
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps == 0:
            video_fps = 30  # Default fallback

        # Calculate frame interval
        frame_interval = max(1, int(video_fps / self.sample_fps))

        frames = []
        frame_count = 0

        while cap.isOpened() and len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

            frame_count += 1

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames extracted from: {video_path}")

        return frames

    def embed(self, video_path: str) -> np.ndarray:
        if not self._is_loaded:
            raise RuntimeError("Model not loaddning")

        frames = self.extract_frames(video_path)

        pil_frames = [Image.fromarray(frame) for frame in frames]

        frame_embeddings = self.image_embedder.batch_embed(pil_frames)

        video_embedding = self._temporal_pool(frame_embeddings)

        video_embedding = video_embedding / np.linalg.norm(video_embedding)

        return video_embedding

    def batch_embed(
        self,
        video_paths: List[str],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        
        embeddings = []
        for video_path in video_paths:
            embedding = self.embed(video_path)
            embeddings.append(embedding)

        return np.array(embeddings)

    def _temporal_pool(self, frame_embeddings: np.ndarray) -> np.ndarray:
        if self.pooling == 'mean':
            return np.mean(frame_embeddings, axis=0)

        elif self.pooling == 'max':
            return np.max(frame_embeddings, axis=0)

        elif self.pooling == 'attention':
            norms = np.linalg.norm(frame_embeddings, axis=1, keepdims=True)
            weights = norms / np.sum(norms)
            return np.sum(frame_embeddings * weights, axis=0)

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

    def warmup(self, sample_input: Optional[str] = None) -> None:
        
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.image_embedder.warmup()
        print("Video embedder warmup complete")


def create_dummy_video(output_path: str, duration_sec: int = 2, fps: int = 10) -> str:
    """Create a dummy test video with changing colors."""
    import tempfile
    import os

    color_sequence = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
    ]

    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    frames_per_color = total_frames // len(color_sequence)

    for color_idx, color in enumerate(color_sequence):
        for _ in range(frames_per_color):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = color[::-1]  # BGR for OpenCV

            # Add text
            cv2.putText(
                frame,
                f"Color {color_idx + 1}",
                (50, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                3,
            )

            out.write(frame)

    out.release()
    return output_path


def test_video_embedder():
    print("=" * 60)
    print("VIDEO EMBEDDER TEST")
    print("=" * 60)

    # Initialize embedder
    print("\n--- Test 1: Load Model ---")
    embedder = VideoEmbedder(device="cpu", sample_fps=2, max_frames=20)
    embedder.load_model()

    # Create test videos
    print("\n--- Test 2: Create Test Videos ---")
    import tempfile
    import os

    temp_dir = tempfile.mkdtemp()
    video1_path = os.path.join(temp_dir, "test_video_1.mp4")
    video2_path = os.path.join(temp_dir, "test_video_2.mp4")

    create_dummy_video(video1_path, duration_sec=2, fps=10)
    create_dummy_video(video2_path, duration_sec=2, fps=10)

    # Test single video embedding
    print("\n--- Test 3: Single Video Embedding ---")
    embedding = embedder.embed(video1_path)
    print(f"✓ Embedding shape: {embedding.shape}")
    print(f"✓ Embedding norm: {np.linalg.norm(embedding):.6f}")
    assert embedding.shape == (512,)
    assert np.abs(np.linalg.norm(embedding) - 1.0) < 1e-5

    # Test different pooling methods
    print("\n--- Test 4: Different Pooling Methods ---")
    pooling_methods = ['mean', 'max', 'attention']
    embeddings = {}

    for method in pooling_methods:
        embedder.pooling = method
        emb = embedder.embed(video1_path)
        embeddings[method] = emb
        print(f"✓ {method.upper()}: norm={np.linalg.norm(emb):.6f}")

    # Test batch embedding
    print("\n--- Test 5: Batch Video Embedding ---")
    embedder.pooling = 'mean'
    batch_embeddings = embedder.batch_embed([video1_path, video2_path])
    print(f"✓ Batch shape: {batch_embeddings.shape}")
    assert batch_embeddings.shape == (2, 512)
    assert np.allclose(np.linalg.norm(batch_embeddings, axis=1), 1.0, atol=1e-5)

    # Cleanup
    print("\n--- Cleanup ---")
    embedder.unload_model()
    os.remove(video1_path)
    os.remove(video2_path)
    os.rmdir(temp_dir)
    print("✓ Cleaned up test files")

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_video_embedder()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
