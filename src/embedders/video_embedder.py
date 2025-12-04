import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Union
from .base_embedder import BaseEmbedder
from .image_embedder import ImageEmbedder


class VideoEmbedder(BaseEmbedder):
    """Video embedder using CLIP via frame sampling and temporal pooling."""

    def __init__(self,
                 model_name: str = "openai/clip-vit-base-patch32",
                 device: str = 'cpu',
                 sample_fps: int = 1,
                 max_frames: int = 100,
                 pooling: str = 'mean'):
        """
        Initialize video embedder.

        Args:
            model_name: CLIP model name
            device: Device to run on
            sample_fps: Frames per second to sample
            max_frames: Maximum number of frames to process
            pooling: Temporal pooling method ('mean', 'max', 'attention')
        """
        super().__init__(model_name, device)
        self.sample_fps = sample_fps
        self.max_frames = max_frames
        self.pooling = pooling
        self.image_embedder = None
        self.embedding_dim = 512  # CLIP base dimension

    def load_model(self):
        """Load CLIP model via ImageEmbedder."""
        print(f"Loading video embedder with CLIP model: {self.model_name}")
        self.image_embedder = ImageEmbedder(self.model_name, self.device)
        self.image_embedder.load_model()
        self.model = self.image_embedder.model
        print("Video embedder loaded")

    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        """
        Extract frames from video at specified fps.

        Args:
            video_path: Path to video file

        Returns:
            List of frames as numpy arrays (RGB)
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # Default fallback

        frame_interval = max(1, int(fps / self.sample_fps))

        frames = []
        frame_count = 0

        while cap.isOpened() and len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

            frame_count += 1

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames extracted from video: {video_path}")

        print(f"Extracted {len(frames)} frames from {video_path}")
        return frames

    def embed(self, video_path: str) -> np.ndarray:
        """
        Generate video embedding via temporal pooling of frame embeddings.

        Args:
            video_path: Path to video file

        Returns:
            Video embedding vector (normalized)
        """
        if self.image_embedder is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Extract frames
        frames = self.extract_frames(video_path)

        # Embed each frame
        frame_embeddings = []

        for frame in frames:
            # Convert numpy array to PIL Image
            pil_frame = Image.fromarray(frame)

            # Get embedding using image embedder
            embedding = self.image_embedder.embed(pil_frame)
            frame_embeddings.append(embedding)

        frame_embeddings = np.array(frame_embeddings)

        # Temporal pooling
        if self.pooling == 'mean':
            video_embedding = np.mean(frame_embeddings, axis=0)
        elif self.pooling == 'max':
            video_embedding = np.max(frame_embeddings, axis=0)
        elif self.pooling == 'attention':
            # Simple attention: weight frames by their norm
            norms = np.linalg.norm(frame_embeddings, axis=1, keepdims=True)
            weights = norms / np.sum(norms)
            video_embedding = np.sum(frame_embeddings * weights, axis=0)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # Normalize
        video_embedding = video_embedding / np.linalg.norm(video_embedding)

        return video_embedding

    def batch_embed(self, video_paths: List[str]) -> np.ndarray:
        """
        Generate embeddings for batch of videos.

        Args:
            video_paths: List of video file paths

        Returns:
            Matrix of video embedding vectors
        """
        embeddings = []

        for video_path in video_paths:
            embedding = self.embed(video_path)
            embeddings.append(embedding)

        return np.array(embeddings)

    def embed_frames_batch(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Embed a batch of frames efficiently.

        Args:
            frames: List of frame arrays

        Returns:
            Matrix of frame embeddings
        """
        if self.image_embedder is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Convert to PIL images
        pil_frames = [Image.fromarray(frame) for frame in frames]

        # Batch embed using image embedder
        return self.image_embedder.batch_embed(pil_frames)
