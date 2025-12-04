import pytest
import numpy as np
import time

from src.utils.metrics import EvaluationMetrics


class TestEvaluationMetrics:
    """Test suite for evaluation metrics."""

    def test_measure_latency(self):
        """Test latency measurement."""
        def slow_function(sleep_time):
            time.sleep(sleep_time)
            return "done"

        result, latency = EvaluationMetrics.measure_latency(slow_function, 0.1)

        assert result == "done"
        assert latency >= 100  # At least 100ms
        assert latency < 200  # But not too much more

    def test_recall_at_k(self):
        """Test Recall@K calculation."""
        retrieved = [1, 2, 3, 4, 5]
        relevant = [2, 4, 6, 8]

        recall = EvaluationMetrics.calculate_recall_at_k(retrieved, relevant, k=5)

        # Retrieved 2 out of 4 relevant items
        assert recall == 0.5

    def test_precision_at_k(self):
        """Test Precision@K calculation."""
        retrieved = [1, 2, 3, 4, 5]
        relevant = [2, 4, 6, 8]

        precision = EvaluationMetrics.calculate_precision_at_k(retrieved, relevant, k=5)

        # 2 relevant out of 5 retrieved
        assert precision == 0.4

    def test_average_precision(self):
        """Test Average Precision calculation."""
        retrieved = [1, 2, 3, 4, 5]
        relevant = [2, 4]

        ap = EvaluationMetrics.calculate_average_precision(retrieved, relevant)

        # AP = (1/2 + 2/4) / 2 = 0.5
        assert ap == 0.5

    def test_map(self):
        """Test Mean Average Precision calculation."""
        all_retrieved = [
            [1, 2, 3],
            [4, 5, 6]
        ]

        all_relevant = [
            [2, 3],
            [4, 7]
        ]

        map_score = EvaluationMetrics.calculate_map(all_retrieved, all_relevant)

        assert 0.0 <= map_score <= 1.0

    def test_ndcg_at_k(self):
        """Test NDCG@K calculation."""
        retrieved = [1, 2, 3, 4, 5]
        relevant = [1, 2, 3]

        ndcg = EvaluationMetrics.calculate_ndcg_at_k(retrieved, relevant, k=5)

        assert 0.0 <= ndcg <= 1.0
        # Perfect ranking for first 3
        assert ndcg > 0.8

    def test_throughput_calculation(self):
        """Test throughput calculation."""
        throughput = EvaluationMetrics.calculate_throughput(
            num_operations=100,
            latency_ms=1000
        )

        # 100 operations in 1 second = 100 ops/sec
        assert throughput == 100.0

    def test_recall_edge_cases(self):
        """Test Recall@K edge cases."""
        # No relevant items
        recall = EvaluationMetrics.calculate_recall_at_k([1, 2, 3], [], k=3)
        assert recall == 0.0

        # All relevant retrieved
        recall = EvaluationMetrics.calculate_recall_at_k(
            [1, 2, 3, 4],
            [1, 2],
            k=4
        )
        assert recall == 1.0

    def test_precision_edge_cases(self):
        """Test Precision@K edge cases."""
        # K = 0
        precision = EvaluationMetrics.calculate_precision_at_k([1, 2, 3], [1], k=0)
        assert precision == 0.0

        # All retrieved are relevant
        precision = EvaluationMetrics.calculate_precision_at_k(
            [1, 2, 3],
            [1, 2, 3],
            k=3
        )
        assert precision == 1.0
