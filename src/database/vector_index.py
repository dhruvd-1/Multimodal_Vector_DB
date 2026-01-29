import hnswlib
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Optional


class VectorIndex:
    """hnswlib-based vector index with FP16 compression."""

    def __init__(self, dimension: int, index_type: str = 'hnsw', metric: str = 'cosine'):
        """
        Initialize vector index.

        Args:
            dimension: Embedding dimension
            index_type: Type of index (only 'hnsw' supported)
            metric: Distance metric ('cosine', 'l2', or 'ip')
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.index = None
        self.metadata = []  # Store metadata for each vector
        self.id_counter = 0
        self._use_fp16 = True

    @property
    def compression_ratio(self) -> float:
        """Return compression ratio for FP16."""
        return 2.0 if self._use_fp16 else 1.0

    def build_index(self, max_elements: int = 100000, ef_construction: int = 128, M: int = 20):
        """
        Build hnswlib index.

        Args:
            max_elements: Maximum number of elements
            ef_construction: Construction time/accuracy trade-off
            M: Number of bi-directional links per element
        """
        print(f"Building {self.index_type} index with dimension {self.dimension}")

        # Map metric names
        space_map = {
            'cosine': 'cosine',
            'l2': 'l2',
            'ip': 'ip'
        }

        if self.metric not in space_map:
            raise ValueError(f"Unknown metric: {self.metric}. Use 'cosine', 'l2', or 'ip'")

        # Initialize hnswlib index
        if self.index_type == 'hnsw':
            self.index = hnswlib.Index(space=space_map[self.metric], dim=self.dimension)
            self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
            self.index.set_ef(50)  # Default ef for search
        elif self.index_type == 'flat':
            self.index = hnswlib.BFIndex(space=space_map[self.metric], dim=self.dimension)
            self.index.init_index(max_elements=max_elements)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}. Use 'hnsw' or 'flat'")

        print(f"Index built successfully")

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        Add vectors with associated metadata (with FP16 compression).

        Args:
            vectors: Vectors to add (shape: [n, dimension])
            metadata: List of metadata dicts for each vector
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Convert to float32 first, then to float16 for compression
        vectors = vectors.astype('float32')

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if len(metadata) != vectors.shape[0]:
            raise ValueError("Number of metadata entries must match number of vectors")

        # Apply FP16 compression
        if self._use_fp16:
            vectors_compressed = vectors.astype('float16').astype('float32')
        else:
            vectors_compressed = vectors

        # Generate IDs for this batch
        ids = np.arange(self.id_counter, self.id_counter + vectors.shape[0])

        # Add vectors to index
        self.index.add_items(vectors_compressed, ids)

        # Add metadata with IDs
        for meta in metadata:
            meta['id'] = self.id_counter
            self.id_counter += 1
            self.metadata.append(meta)

        print(f"Added {vectors.shape[0]} vectors. Total: {self.index.get_current_count()}")

    def search(self, query_vector: np.ndarray, k: int = 10,
              filter_fn: Optional[callable] = None) -> List[Dict[str, Any]]:
        """
        Search for k nearest neighbors.

        Args:
            query_vector: Query vector
            k: Number of results to return
            filter_fn: Optional function to filter results by metadata

        Returns:
            List of dicts with 'distance', 'metadata', and 'id' keys
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Apply FP16 compression to query
        query_vector = query_vector.astype('float32').reshape(-1)
        if self._use_fp16:
            query_vector = query_vector.astype('float16').astype('float32')

        # Search for more results if filtering
        search_k = min(k * 10 if filter_fn else k, self.index.get_current_count())

        labels, distances = self.index.knn_query(query_vector, k=search_k)

        results = []
        for idx, dist in zip(labels[0], distances[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            # Convert distance to similarity
            if self.metric == 'cosine':
                similarity = float(1 - dist)
            elif self.metric == 'l2':
                similarity = float(1 / (1 + dist))
            else:  # ip
                similarity = float(dist)

            result = {
                'distance': float(dist),
                'similarity': similarity,
                'metadata': self.metadata[idx].copy(),
                'id': self.metadata[idx]['id']
            }

            # Apply filter if provided
            if filter_fn is None or filter_fn(result['metadata']):
                results.append(result)

            if len(results) >= k:
                break

        return results

    def search_batch(self, query_vectors: np.ndarray, k: int = 10) -> List[List[Dict[str, Any]]]:
        """
        Batch search for multiple queries.

        Args:
            query_vectors: Multiple query vectors (shape: [n, dimension])
            k: Number of results per query

        Returns:
            List of result lists, one per query
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_vectors = query_vectors.astype('float32')
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)

        # Apply FP16 compression
        if self._use_fp16:
            query_vectors = query_vectors.astype('float16').astype('float32')

        labels, distances = self.index.knn_query(query_vectors, k=k)

        all_results = []
        for i in range(len(query_vectors)):
            results = []
            for idx, dist in zip(labels[i], distances[i]):
                if idx < 0 or idx >= len(self.metadata):
                    continue

                # Convert distance to similarity
                if self.metric == 'cosine':
                    similarity = float(1 - dist)
                elif self.metric == 'l2':
                    similarity = float(1 / (1 + dist))
                else:  # ip
                    similarity = float(dist)

                results.append({
                    'distance': float(dist),
                    'similarity': similarity,
                    'metadata': self.metadata[idx].copy(),
                    'id': self.metadata[idx]['id']
                })

            all_results.append(results)

        return all_results

    def remove_vectors(self, ids: List[int]):
        """
        Remove vectors by ID (marking as deleted in metadata).

        Args:
            ids: List of IDs to remove
        """
        # Note: hnswlib doesn't support true deletion, so we mark in metadata
        for meta in self.metadata:
            if meta['id'] in ids:
                meta['deleted'] = True

    def save(self, path: str):
        """
        Save index and metadata to disk.

        Args:
            path: Base path for saving (without extension)
        """
        if self.index is None:
            raise RuntimeError("Index not built. Cannot save.")

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        # Save hnswlib index
        self.index.save_index(f"{path}.index")

        # Save metadata and config
        config = {
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metric': self.metric,
            'id_counter': self.id_counter,
            'metadata': self.metadata,
            'use_fp16': self._use_fp16
        }

        with open(f"{path}.metadata", 'wb') as f:
            pickle.dump(config, f)

        print(f"Index saved to {path}")

    def load(self, path: str):
        """
        Load index and metadata from disk.

        Args:
            path: Base path for loading (without extension)
        """
        # Load metadata and config first
        with open(f"{path}.metadata", 'rb') as f:
            config = pickle.load(f)

        self.dimension = config['dimension']
        self.index_type = config['index_type']
        self.metric = config['metric']
        self.id_counter = config['id_counter']
        self.metadata = config['metadata']
        self._use_fp16 = config.get('use_fp16', True)

        # Initialize and load hnswlib index
        space_map = {'cosine': 'cosine', 'l2': 'l2', 'ip': 'ip'}
        self.index = hnswlib.Index(space=space_map[self.metric], dim=self.dimension)
        self.index.load_index(f"{path}.index")

        print(f"Index loaded from {path}. Contains {self.index.get_current_count()} vectors")

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if self.index is None:
            return {'status': 'not_built'}

        return {
            'total_vectors': self.index.get_current_count(),
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metric': self.metric,
            'compression_ratio': self.compression_ratio,
            'use_fp16': self._use_fp16
        }
