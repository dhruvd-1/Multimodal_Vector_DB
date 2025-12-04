import time
import numpy as np
from typing import List, Dict, Any, Callable
from functools import wraps


class EvaluationMetrics:
    """Evaluation metrics for the vector database system."""

    @staticmethod
    def measure_latency(func: Callable, *args, **kwargs) -> tuple:
        """
        Measure function execution time.

        Args:
            func: Function to measure
            *args, **kwargs: Function arguments

        Returns:
            Tuple of (result, latency_in_ms)
        """
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000

        return result, latency_ms

    @staticmethod
    def calculate_recall_at_k(retrieved_ids: List[int],
                              relevant_ids: List[int],
                              k: int) -> float:
        """
        Calculate Recall@K metric.

        Args:
            retrieved_ids: List of retrieved item IDs
            relevant_ids: List of ground truth relevant IDs
            k: Top-k cutoff

        Returns:
            Recall@K score (0-1)
        """
        if len(relevant_ids) == 0:
            return 0.0

        retrieved_k = set(retrieved_ids[:k])
        relevant = set(relevant_ids)

        intersection = len(retrieved_k & relevant)

        return intersection / len(relevant)

    @staticmethod
    def calculate_precision_at_k(retrieved_ids: List[int],
                                 relevant_ids: List[int],
                                 k: int) -> float:
        """
        Calculate Precision@K metric.

        Args:
            retrieved_ids: List of retrieved item IDs
            relevant_ids: List of ground truth relevant IDs
            k: Top-k cutoff

        Returns:
            Precision@K score (0-1)
        """
        if k == 0:
            return 0.0

        retrieved_k = set(retrieved_ids[:k])
        relevant = set(relevant_ids)

        intersection = len(retrieved_k & relevant)

        return intersection / k

    @staticmethod
    def calculate_average_precision(retrieved_ids: List[int],
                                    relevant_ids: List[int]) -> float:
        """
        Calculate Average Precision (AP).

        Args:
            retrieved_ids: List of retrieved item IDs (ordered)
            relevant_ids: List of ground truth relevant IDs

        Returns:
            Average Precision score (0-1)
        """
        relevant_set = set(relevant_ids)

        if len(relevant_set) == 0:
            return 0.0

        precisions = []
        num_relevant = 0

        for k, item_id in enumerate(retrieved_ids, start=1):
            if item_id in relevant_set:
                num_relevant += 1
                precision_at_k = num_relevant / k
                precisions.append(precision_at_k)

        if len(precisions) == 0:
            return 0.0

        return sum(precisions) / len(relevant_set)

    @staticmethod
    def calculate_map(all_retrieved: List[List[int]],
                     all_relevant: List[List[int]]) -> float:
        """
        Calculate Mean Average Precision (MAP).

        Args:
            all_retrieved: List of retrieved ID lists (one per query)
            all_relevant: List of relevant ID lists (one per query)

        Returns:
            MAP score (0-1)
        """
        if len(all_retrieved) != len(all_relevant):
            raise ValueError("Number of queries must match")

        aps = []
        for retrieved, relevant in zip(all_retrieved, all_relevant):
            ap = EvaluationMetrics.calculate_average_precision(retrieved, relevant)
            aps.append(ap)

        return np.mean(aps) if aps else 0.0

    @staticmethod
    def calculate_ndcg_at_k(retrieved_ids: List[int],
                           relevant_ids: List[int],
                           k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).

        Args:
            retrieved_ids: List of retrieved item IDs
            relevant_ids: List of ground truth relevant IDs
            k: Top-k cutoff

        Returns:
            NDCG@K score (0-1)
        """
        relevant_set = set(relevant_ids)

        # Calculate DCG
        dcg = 0.0
        for i, item_id in enumerate(retrieved_ids[:k], start=1):
            if item_id in relevant_set:
                dcg += 1.0 / np.log2(i + 1)

        # Calculate ideal DCG
        idcg = 0.0
        for i in range(min(k, len(relevant_ids))):
            idcg += 1.0 / np.log2(i + 2)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def measure_storage_per_vector(index, num_vectors: int) -> float:
        """
        Estimate storage per vector in KB.

        Args:
            index: FAISS index
            num_vectors: Number of vectors in index

        Returns:
            Storage per vector in KB
        """
        import faiss
        import tempfile
        import os

        if num_vectors == 0:
            return 0.0

        # Save index to temp file and measure size
        temp_path = tempfile.mktemp(suffix='.index')

        try:
            faiss.write_index(index, temp_path)
            size_bytes = os.path.getsize(temp_path)
            os.remove(temp_path)

            # KB per vector
            return (size_bytes / num_vectors) / 1024

        except Exception as e:
            print(f"Error measuring storage: {e}")
            return 0.0

    @staticmethod
    def calculate_throughput(num_operations: int, latency_ms: float) -> float:
        """
        Calculate throughput (operations per second).

        Args:
            num_operations: Number of operations performed
            latency_ms: Total latency in milliseconds

        Returns:
            Throughput in ops/second
        """
        if latency_ms <= 0:
            return 0.0

        latency_sec = latency_ms / 1000
        return num_operations / latency_sec

    @staticmethod
    def benchmark_search(search_func: Callable,
                        queries: List,
                        k: int = 10,
                        num_runs: int = 10) -> Dict[str, float]:
        """
        Benchmark search performance.

        Args:
            search_func: Search function to benchmark
            queries: List of queries
            k: Number of results per query
            num_runs: Number of runs for averaging

        Returns:
            Dictionary with benchmark results
        """
        latencies = []

        for _ in range(num_runs):
            for query in queries:
                _, latency = EvaluationMetrics.measure_latency(
                    search_func, query, k=k
                )
                latencies.append(latency)

        return {
            'mean_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'throughput_qps': 1000 / np.mean(latencies),
        }


def timer(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result, latency = EvaluationMetrics.measure_latency(func, *args, **kwargs)
        print(f"{func.__name__} took {latency:.2f}ms")
        return result
    return wrapper
