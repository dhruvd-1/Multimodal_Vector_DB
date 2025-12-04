import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Optional


class VectorIndex:
    """FAISS-based vector index supporting multiple index types."""

    def __init__(self, dimension: int, index_type: str = 'flat', metric: str = 'l2'):
        """
        Initialize vector index.

        Args:
            dimension: Embedding dimension
            index_type: Type of FAISS index ('flat', 'ivf', 'pq', 'hnsw')
            metric: Distance metric ('l2' or 'ip' for inner product)
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.index = None
        self.metadata = []  # Store metadata for each vector
        self.id_counter = 0

    def build_index(self, nlist: int = 100, m: int = 64, nbits: int = 8,
                   hnsw_m: int = 32):
        """
        Build FAISS index.

        Args:
            nlist: Number of clusters for IVF (used with 'ivf')
            m: Number of subquantizers for PQ (used with 'pq')
            nbits: Bits per subquantizer for PQ (used with 'pq')
            hnsw_m: Number of neighbors for HNSW (used with 'hnsw')
        """
        print(f"Building {self.index_type} index with dimension {self.dimension}")

        if self.index_type == 'flat':
            # Exact search - fastest for small datasets
            if self.metric == 'l2':
                self.index = faiss.IndexFlatL2(self.dimension)
            else:
                self.index = faiss.IndexFlatIP(self.dimension)

        elif self.index_type == 'ivf':
            # Inverted file index - faster approximate search
            if self.metric == 'l2':
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            else:
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)

        elif self.index_type == 'pq':
            # Product quantization - compressed storage
            self.index = faiss.IndexPQ(self.dimension, m, nbits)

        elif self.index_type == 'hnsw':
            # Hierarchical navigable small world - graph-based fast search
            if self.metric == 'l2':
                self.index = faiss.IndexHNSWFlat(self.dimension, hnsw_m)
            else:
                self.index = faiss.IndexHNSWFlat(self.dimension, hnsw_m, faiss.METRIC_INNER_PRODUCT)

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        print(f"Index built successfully")

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        Add vectors with associated metadata.

        Args:
            vectors: Vectors to add (shape: [n, dimension])
            metadata: List of metadata dicts for each vector
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        vectors = vectors.astype('float32')

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if len(metadata) != vectors.shape[0]:
            raise ValueError("Number of metadata entries must match number of vectors")

        # Train index if needed (for IVF, PQ)
        if self.index_type in ['ivf', 'pq'] and not self.index.is_trained:
            print(f"Training index with {vectors.shape[0]} vectors...")
            self.index.train(vectors)
            print("Training complete")

        # Add vectors
        self.index.add(vectors)

        # Add metadata with IDs
        for meta in metadata:
            meta['id'] = self.id_counter
            self.id_counter += 1
            self.metadata.append(meta)

        print(f"Added {vectors.shape[0]} vectors. Total: {self.index.ntotal}")

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

        query_vector = query_vector.astype('float32').reshape(1, -1)

        # Search for more results if filtering
        search_k = k * 10 if filter_fn else k

        distances, indices = self.index.search(query_vector, search_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            result = {
                'distance': float(dist),
                'similarity': float(1 / (1 + dist)) if self.metric == 'l2' else float(dist),
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

        distances, indices = self.index.search(query_vectors, k)

        all_results = []
        for i in range(len(query_vectors)):
            results = []
            for dist, idx in zip(distances[i], indices[i]):
                if idx < 0 or idx >= len(self.metadata):
                    continue

                results.append({
                    'distance': float(dist),
                    'similarity': float(1 / (1 + dist)) if self.metric == 'l2' else float(dist),
                    'metadata': self.metadata[idx].copy(),
                    'id': self.metadata[idx]['id']
                })

            all_results.append(results)

        return all_results

    def remove_vectors(self, ids: List[int]):
        """
        Remove vectors by ID (only supported for some index types).

        Args:
            ids: List of IDs to remove
        """
        # Note: FAISS doesn't support true deletion, so we'll mark in metadata
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

        # Save FAISS index
        faiss.write_index(self.index, f"{path}.index")

        # Save metadata and config
        config = {
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metric': self.metric,
            'id_counter': self.id_counter,
            'metadata': self.metadata
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
        # Load FAISS index
        self.index = faiss.read_index(f"{path}.index")

        # Load metadata and config
        with open(f"{path}.metadata", 'rb') as f:
            config = pickle.load(f)

        self.dimension = config['dimension']
        self.index_type = config['index_type']
        self.metric = config['metric']
        self.id_counter = config['id_counter']
        self.metadata = config['metadata']

        print(f"Index loaded from {path}. Contains {self.index.ntotal} vectors")

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if self.index is None:
            return {'status': 'not_built'}

        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metric': self.metric,
            'is_trained': self.index.is_trained if hasattr(self.index, 'is_trained') else True,
        }
