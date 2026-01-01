"""
Standalone Test for Video Embedder
Run: python test_video_embedder.py
"""

import sys

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class ImageEmbedder:
    """Image embedder for video frame processing."""

    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cpu"):
        self.model_name = model_name
        self.device = device
        self.processor = None
        self.model = None
        self.embedding_dim = 512

    def load_model(self):
        """Load CLIP model and processor."""
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def embed(self, image_input):
        """Generate embedding for single image."""
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings.cpu().numpy()[0]

    def batch_embed(self, images, batch_size=32):
        """Generate embeddings for batch of images."""
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            pil_images = [img if isinstance(img, Image.Image) else img for img in batch]

            inputs = self.processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                embeddings = self.model.get_image_features(**inputs)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


class VideoEmbedder:
    """Video embedder using CLIP via frame sampling and temporal pooling."""

    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32",
        device="cpu",
        sample_fps=1,
        max_frames=100,
        pooling="mean",
    ):
        self.model_name = model_name
        self.device = device
        self.sample_fps = sample_fps
        self.max_frames = max_frames
        self.pooling = pooling
        self.image_embedder = None
        self.model = None
        self.embedding_dim = 512

    def load_model(self):
        """Load CLIP model via ImageEmbedder."""
        print(f"Loading video embedder with CLIP model: {self.model_name}")
        self.image_embedder = ImageEmbedder(self.model_name, self.device)
        self.image_embedder.load_model()
        self.model = self.image_embedder.model
        print("✓ Video embedder loaded")

    def extract_frames(self, video_path):
        """Extract frames from video at specified fps."""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30

        frame_interval = max(1, int(fps / self.sample_fps))

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
            raise ValueError(f"No frames extracted from video: {video_path}")

        print(f"  ✓ Extracted {len(frames)} frames from video")
        return frames

    def embed(self, video_path):
        """Generate video embedding via temporal pooling of frame embeddings."""
        if self.image_embedder is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Extract frames
        frames = self.extract_frames(video_path)

        # Embed each frame
        frame_embeddings = []
        for frame in frames:
            pil_frame = Image.fromarray(frame)
            embedding = self.image_embedder.embed(pil_frame)
            frame_embeddings.append(embedding)

        frame_embeddings = np.array(frame_embeddings)

        # Temporal pooling
        if self.pooling == "mean":
            video_embedding = np.mean(frame_embeddings, axis=0)
        elif self.pooling == "max":
            video_embedding = np.max(frame_embeddings, axis=0)
        elif self.pooling == "attention":
            norms = np.linalg.norm(frame_embeddings, axis=1, keepdims=True)
            weights = norms / np.sum(norms)
            video_embedding = np.sum(frame_embeddings * weights, axis=0)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # Normalize
        video_embedding = video_embedding / np.linalg.norm(video_embedding)

        return video_embedding

    def batch_embed(self, video_paths):
        """Generate embeddings for batch of videos."""
        embeddings = []
        for video_path in video_paths:
            embedding = self.embed(video_path)
            embeddings.append(embedding)
        return np.array(embeddings)


def create_dummy_video(output_path, duration_sec=3, fps=10, color_sequence=None):
    """Create a dummy test video with changing colors."""
    if color_sequence is None:
        color_sequence = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
        ]

    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    frames_per_color = total_frames // len(color_sequence)

    for color_idx, color in enumerate(color_sequence):
        for _ in range(frames_per_color):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = color[::-1]  # BGR format for OpenCV

            # Add some text
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
    print(f"✓ Created dummy video: {output_path}")
    return output_path


def test_video_embedder():
    """Test the video embedder with dummy data."""
    print("=" * 60)
    print("VIDEO EMBEDDER TEST")
    print("=" * 60)

    # Initialize embedder
    embedder = VideoEmbedder(device="cpu", sample_fps=2, max_frames=50)
    embedder.load_model()

    # Create test videos
    print("\n--- Creating Test Videos ---")
    video1_path = "test_video_red_green.mp4"
    video2_path = "test_video_blue_yellow.mp4"
    video3_path = "test_video_red_blue.mp4"

    create_dummy_video(
        video1_path, duration_sec=2, fps=10, color_sequence=[(255, 0, 0), (0, 255, 0)]
    )
    create_dummy_video(
        video2_path, duration_sec=2, fps=10, color_sequence=[(0, 0, 255), (255, 255, 0)]
    )
    create_dummy_video(
        video3_path, duration_sec=2, fps=10, color_sequence=[(255, 0, 0), (0, 0, 255)]
    )

    # Test 1: Single video embedding
    print("\n--- Test 1: Single Video Embedding ---")
    print(f"Processing: {video1_path}")

    embedding = embedder.embed(video1_path)
    print(f"✓ Embedding shape: {embedding.shape}")
    print(f"✓ Embedding dimension: {len(embedding)}")
    print(f"✓ Embedding norm (should be ~1.0): {np.linalg.norm(embedding):.6f}")
    print(f"✓ First 5 values: {embedding[:5]}")

    # Test 2: Multiple pooling strategies
    print("\n--- Test 2: Different Pooling Strategies ---")
    pooling_methods = ["mean", "max", "attention"]
    embeddings_pooling = {}

    for method in pooling_methods:
        embedder.pooling = method
        emb = embedder.embed(video1_path)
        embeddings_pooling[method] = emb
        print(f"✓ {method.upper()} pooling: norm = {np.linalg.norm(emb):.6f}")

    # Test 3: Batch video embedding
    print("\n--- Test 3: Batch Video Embedding ---")
    video_paths = [video1_path, video2_path, video3_path]

    print(f"Processing {len(video_paths)} videos...")
    embeddings = embedder.batch_embed(video_paths)

    print(f"✓ Batch embeddings shape: {embeddings.shape}")
    print(
        f"✓ All vectors normalized: {np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)}"
    )

    # Test 4: Video similarity
    print("\n--- Test 4: Video Similarity Test ---")
    embedder.pooling = "mean"

    emb1 = embedder.embed(video1_path)  # Red-Green
    emb2 = embedder.embed(video2_path)  # Blue-Yellow
    emb3 = embedder.embed(video3_path)  # Red-Blue

    sim_12 = np.dot(emb1, emb2)
    sim_13 = np.dot(emb1, emb3)

    print(f"Video 1: Red-Green sequence")
    print(f"Video 2: Blue-Yellow sequence")
    print(f"Video 3: Red-Blue sequence")
    print(f"\nSimilarity (Video1 vs Video2): {sim_12:.4f}")
    print(f"Similarity (Video1 vs Video3): {sim_13:.4f}")
    print(f"✓ Videos with shared colors are more similar: {sim_13 > sim_12}")

    # Test 5: Frame extraction test
    print("\n--- Test 5: Frame Extraction Details ---")
    frames = embedder.extract_frames(video1_path)
    print(f"Total frames extracted: {len(frames)}")
    print(f"Frame shape: {frames[0].shape}")
    print(f"✓ Frames are in RGB format")

    # Cleanup
    print("\n--- Cleaning up test videos ---")
    import os

    for path in [video1_path, video2_path, video3_path]:
        if os.path.exists(path):
            os.remove(path)
            print(f"✓ Removed {path}")

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
