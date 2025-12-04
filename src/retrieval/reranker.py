import numpy as np
from typing import List, Dict, Any, Callable


class Reranker:
    """Rerank search results using various strategies."""

    def __init__(self, strategy: str = 'distance'):
        """
        Initialize reranker.

        Args:
            strategy: Reranking strategy ('distance', 'diversity', 'combined')
        """
        self.strategy = strategy

    def rerank(self,
              results: List[Dict[str, Any]],
              query_embedding: np.ndarray = None,
              top_k: int = None) -> List[Dict[str, Any]]:
        """
        Rerank search results.

        Args:
            results: Initial search results
            query_embedding: Optional query embedding for advanced reranking
            top_k: Number of top results to return

        Returns:
            Reranked results
        """
        if self.strategy == 'distance':
            # Already sorted by distance
            reranked = results

        elif self.strategy == 'diversity':
            # Maximal Marginal Relevance (MMR)
            reranked = self._mmr_rerank(results, query_embedding)

        elif self.strategy == 'combined':
            # Combine distance and diversity
            reranked = self._combined_rerank(results, query_embedding)

        else:
            reranked = results

        if top_k:
            return reranked[:top_k]

        return reranked

    def _mmr_rerank(self,
                   results: List[Dict[str, Any]],
                   query_embedding: np.ndarray,
                   lambda_param: float = 0.5) -> List[Dict[str, Any]]:
        """
        Maximal Marginal Relevance reranking for diversity.

        Args:
            results: Initial results
            query_embedding: Query embedding
            lambda_param: Trade-off between relevance and diversity (0-1)

        Returns:
            Reranked results
        """
        if len(results) <= 1 or query_embedding is None:
            return results

        # Extract embeddings (assume they're stored in metadata)
        embeddings = []
        for r in results:
            if 'embedding' in r['metadata']:
                embeddings.append(r['metadata']['embedding'])
            else:
                # If no embedding, keep original order
                return results

        embeddings = np.array(embeddings)

        # MMR algorithm
        selected = []
        remaining = list(range(len(results)))

        while remaining and len(selected) < len(results):
            if not selected:
                # Select first based on distance
                idx = remaining[0]
            else:
                # Compute MMR score for each remaining
                mmr_scores = []
                for idx in remaining:
                    # Relevance: similarity to query
                    relevance = self._cosine_similarity(
                        query_embedding,
                        embeddings[idx]
                    )

                    # Diversity: max similarity to selected
                    max_sim = max([
                        self._cosine_similarity(embeddings[idx], embeddings[s])
                        for s in selected
                    ])

                    # MMR score
                    mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                    mmr_scores.append(mmr)

                # Select highest MMR
                best_idx = remaining[np.argmax(mmr_scores)]
                idx = best_idx

            selected.append(idx)
            remaining.remove(idx)

        # Reorder results
        return [results[i] for i in selected]

    def _combined_rerank(self,
                        results: List[Dict[str, Any]],
                        query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """
        Combine distance and diversity scores.

        Args:
            results: Initial results
            query_embedding: Query embedding

        Returns:
            Reranked results
        """
        # Apply MMR with balanced parameters
        return self._mmr_rerank(results, query_embedding, lambda_param=0.7)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    def filter_results(self,
                      results: List[Dict[str, Any]],
                      filter_fn: Callable) -> List[Dict[str, Any]]:
        """
        Filter results based on custom function.

        Args:
            results: Search results
            filter_fn: Function that takes metadata and returns bool

        Returns:
            Filtered results
        """
        return [r for r in results if filter_fn(r['metadata'])]
