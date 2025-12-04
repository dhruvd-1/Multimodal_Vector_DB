import os
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import numpy as np

from ..embedders import TextEmbedder, ImageEmbedder, VideoEmbedder, AudioEmbedder
from ..database import VectorIndex, StorageManager


class MultiModalSearchEngine:
    """Unified search engine for multimodal content."""

    def __init__(self,
                 embedding_dim: int = 512,
                 model_name: str = "openai/clip-vit-base-patch32",
                 device: str = 'cpu',
                 index_type: str = 'flat',
                 storage_path: Optional[str] = None):
        """
        Initialize multimodal search engine.

        Args:
            embedding_dim: Dimension of embeddings
            model_name: Model name for embedders
            device: Device to run on
            index_type: Type of FAISS index
            storage_path: Path for persistent storage
        """
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.device = device
        self.index_type = index_type

        # Initialize embedders (lazy loading)
        self.text_embedder = None
        self.image_embedder = None
        self.video_embedder = None
        self.audio_embedder = None

        # Initialize vector index
        self.vector_index = VectorIndex(
            dimension=embedding_dim,
            index_type=index_type,
            metric='l2'
        )

        # Storage manager
        self.storage = StorageManager(storage_path) if storage_path else None

    def initialize(self, modalities: List[str] = ['text', 'image']):
        """
        Load models and build index.

        Args:
            modalities: List of modalities to enable ['text', 'image', 'video', 'audio']
        """
        print(f"Initializing search engine for modalities: {modalities}")

        # Load embedders as needed
        if 'text' in modalities:
            self.text_embedder = TextEmbedder(self.model_name, self.device)
            self.text_embedder.load_model()

        if 'image' in modalities:
            self.image_embedder = ImageEmbedder(self.model_name, self.device)
            self.image_embedder.load_model()

        if 'video' in modalities:
            self.video_embedder = VideoEmbedder(self.model_name, self.device)
            self.video_embedder.load_model()

        if 'audio' in modalities:
            self.audio_embedder = AudioEmbedder(self.model_name, self.device)
            self.audio_embedder.load_model()

        # Build index
        self.vector_index.build_index()

        print("Search engine initialized")

    def ingest_content(self,
                      content: Union[str, Image.Image],
                      content_type: str,
                      metadata: Optional[Dict[str, Any]] = None):
        """
        Ingest and index content.

        Args:
            content: Content to ingest (text string, image path/PIL, video path, audio path)
            content_type: Type of content ('text', 'image', 'video', 'audio')
            metadata: Additional metadata to store
        """
        if metadata is None:
            metadata = {}

        # Generate embedding based on content type
        if content_type == 'text':
            if self.text_embedder is None:
                raise RuntimeError("Text embedder not initialized")
            embedding = self.text_embedder.embed(content)

        elif content_type == 'image':
            if self.image_embedder is None:
                raise RuntimeError("Image embedder not initialized")
            embedding = self.image_embedder.embed(content)

        elif content_type == 'video':
            if self.video_embedder is None:
                raise RuntimeError("Video embedder not initialized")
            embedding = self.video_embedder.embed(content)

        elif content_type == 'audio':
            if self.audio_embedder is None:
                raise RuntimeError("Audio embedder not initialized")
            embedding = self.audio_embedder.embed(content)

        else:
            raise ValueError(f"Unknown content type: {content_type}")

        # Add metadata
        metadata['content_type'] = content_type
        if isinstance(content, str):
            metadata['content_path'] = content
        else:
            metadata['content'] = str(content)

        # Add to index
        self.vector_index.add_vectors(
            embedding.reshape(1, -1),
            [metadata]
        )

    def batch_ingest(self,
                    contents: List[Union[str, Image.Image]],
                    content_type: str,
                    metadata_list: Optional[List[Dict[str, Any]]] = None):
        """
        Batch ingest content for efficiency.

        Args:
            contents: List of content items
            content_type: Type of all content
            metadata_list: List of metadata dicts
        """
        if metadata_list is None:
            metadata_list = [{} for _ in contents]

        # Generate embeddings in batch
        if content_type == 'text':
            if self.text_embedder is None:
                raise RuntimeError("Text embedder not initialized")
            embeddings = self.text_embedder.batch_embed(contents)

        elif content_type == 'image':
            if self.image_embedder is None:
                raise RuntimeError("Image embedder not initialized")
            embeddings = self.image_embedder.batch_embed(contents)

        else:
            # Fall back to individual ingestion
            for content, metadata in zip(contents, metadata_list):
                self.ingest_content(content, content_type, metadata)
            return

        # Add metadata
        for i, metadata in enumerate(metadata_list):
            metadata['content_type'] = content_type
            if isinstance(contents[i], str):
                metadata['content_path'] = contents[i]

        # Add to index
        self.vector_index.add_vectors(embeddings, metadata_list)

    def search(self,
              query: Union[str, Image.Image],
              query_type: str = 'text',
              k: int = 10,
              filter_content_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Cross-modal search.

        Args:
            query: Query (text string or image)
            query_type: Type of query ('text' or 'image')
            k: Number of results
            filter_content_type: Optionally filter by content type

        Returns:
            List of search results with metadata and scores
        """
        # Embed query
        if query_type == 'text':
            if self.text_embedder is None:
                raise RuntimeError("Text embedder not initialized")
            query_embedding = self.text_embedder.embed(query)

        elif query_type == 'image':
            if self.image_embedder is None:
                raise RuntimeError("Image embedder not initialized")
            query_embedding = self.image_embedder.embed(query)

        elif query_type == 'video':
            if self.video_embedder is None:
                raise RuntimeError("Video embedder not initialized")
            query_embedding = self.video_embedder.embed(query)

        elif query_type == 'audio':
            if self.audio_embedder is None:
                raise RuntimeError("Audio embedder not initialized")
            query_embedding = self.audio_embedder.embed(query)

        else:
            raise ValueError(f"Unknown query type: {query_type}")

        # Filter function
        filter_fn = None
        if filter_content_type:
            filter_fn = lambda meta: meta.get('content_type') == filter_content_type

        # Search
        results = self.vector_index.search(query_embedding, k=k, filter_fn=filter_fn)

        return results

    def save(self, path: Optional[str] = None):
        """Save index and configuration."""
        if path is None:
            if self.storage is None:
                raise ValueError("No storage path specified")
            path = self.storage.get_index_path()

        self.vector_index.save(path)

        # Save config
        if self.storage:
            config = {
                'embedding_dim': self.embedding_dim,
                'model_name': self.model_name,
                'index_type': self.index_type,
            }
            self.storage.save_config(config)

    def load(self, path: Optional[str] = None):
        """Load index and configuration."""
        if path is None:
            if self.storage is None:
                raise ValueError("No storage path specified")
            path = self.storage.get_index_path()

        self.vector_index.load(path)

        # Load config
        if self.storage:
            config = self.storage.load_config()
            if config:
                self.embedding_dim = config['embedding_dim']
                self.model_name = config['model_name']
                self.index_type = config['index_type']

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        stats = self.vector_index.get_stats()
        stats['enabled_embedders'] = []

        if self.text_embedder:
            stats['enabled_embedders'].append('text')
        if self.image_embedder:
            stats['enabled_embedders'].append('image')
        if self.video_embedder:
            stats['enabled_embedders'].append('video')
        if self.audio_embedder:
            stats['enabled_embedders'].append('audio')

        return stats
