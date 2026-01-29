from .base_embedder import BaseEmbedder
from .text_embedder import TextEmbedder
from .image_embedder import ImageEmbedder

# Optional imports for heavy dependencies
try:
    from .video_embedder import VideoEmbedder
except ImportError:
    VideoEmbedder = None

try:
    from .audio_embedder import AudioEmbedder
except ImportError:
    AudioEmbedder = None

__all__ = [
    'BaseEmbedder',
    'TextEmbedder',
    'ImageEmbedder',
    'VideoEmbedder',
    'AudioEmbedder',
]
