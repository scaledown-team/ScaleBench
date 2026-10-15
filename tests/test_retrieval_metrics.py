"""Unit tests for retrieval metrics."""
import pytest
from metrics.retrieval import (
    compute_retrieval_precision,
    compute_retrieval_recall,
    compute_retrieval_f1,
    compute_all_retrieval_metrics,
    compute_combined_score,
)


class TestRetrievalPrecision:
    """Tests for compute_retrieval_precision."""
    
    def test_perfect_precision(self):
        """Test precision with perfect predictions."""
        predictions = [[0, 1], [2, 3]]
        references = [[0, 1], [2, 3]]
        
        result = compute_retrieval_precision(predictions, references)
        assert result["retrieval_precision"] == 1.0
    
    def test_zero_precision(self):
        """Test precision with no overlap."""
        predictions = [[0, 1]]
        references = [[2, 3]]
        
        result = compute_retrieval_precision(predictions, references)
        assert result["retrieval_precision"] == 0.0
    
    def test_partial_precision(self):
        """Test precision with partial overlap."""
        # Predicted [0, 1, 2], gold [0, 1]: precision = 2/3
        predictions = [[0, 1, 2]]
        references = [[0, 1]]
        
        result = compute_retrieval_precision(predictions, references)
        assert abs(result["retrieval_precision"] - 0.6667) < 0.01
    
    def test_empty_prediction_zero_precision(self):
        """Test that empty prediction gives zero precision."""
        predictions = [[]]
        references = [[0, 1]]
        
        result = compute_retrieval_precision(predictions, references)
        assert result["retrieval_precision"] == 0.0
    
    def test_none_prediction_zero_precision(self):
        """Test that None prediction gives zero precision."""
        predictions = [None]
        references = [[0, 1]]
        
        result = compute_retrieval_precision(predictions, references)
        assert result["retrieval_precision"] == 0.0
    
    def test_multiple_examples_averaged(self):
        """Test that multiple examples are averaged."""
        # Example 1: precision = 1.0 (perfect)
        # Example 2: precision = 0.5 ([0] out of [0, 1])
        predictions = [[0, 1], [0, 2]]
        references = [[0, 1], [0, 1]]
        
        result = compute_retrieval_precision(predictions, references)
        assert abs(result["retrieval_precision"] - 0.75) < 0.01


class TestRetrievalRecall:
    """Tests for compute_retrieval_recall."""
    
    def test_perfect_recall(self):
        """Test recall with perfect predictions."""
        predictions = [[0, 1], [2, 3]]
        references = [[0, 1], [2, 3]]
        
        result = compute_retrieval_recall(predictions, references)
        assert result["retrieval_recall"] == 1.0
    
    def test_zero_recall(self):
        """Test recall with no overlap."""
        predictions = [[0, 1]]
        references = [[2, 3]]
        
        result = compute_retrieval_recall(predictions, references)
        assert result["retrieval_recall"] == 0.0
    
    def test_partial_recall(self):
        """Test recall with partial overlap."""
        # Predicted [0], gold [0, 1, 2]: recall = 1/3
        predictions = [[0]]
        references = [[0, 1, 2]]
        
        result = compute_retrieval_recall(predictions, references)
        assert abs(result["retrieval_recall"] - 0.3333) < 0.01
    
    def test_empty_gold_full_recall(self):
        """Test that empty gold gives full recall (nothing to find)."""
        predictions = [[0, 1]]
        references = [[]]
        
        result = compute_retrieval_recall(predictions, references)
        assert result["retrieval_recall"] == 1.0
    
    def test_empty_prediction_zero_recall(self):
        """Test that empty prediction gives zero recall when gold exists."""
        predictions = [[]]
        references = [[0, 1]]
        
        result = compute_retrieval_recall(predictions, references)
        assert result["retrieval_recall"] == 0.0


class TestRetrievalF1:
    """Tests for compute_retrieval_f1."""
    
    def test_perfect_f1(self):
        """Test F1 with perfect predictions."""
        predictions = [[0, 1]]
        references = [[0, 1]]
        
        result = compute_retrieval_f1(predictions, references)
        assert result["retrieval_f1"] == 1.0
    
    def test_zero_f1(self):
        """Test F1 with no overlap."""
        predictions = [[0, 1]]
        references = [[2, 3]]
        
        result = compute_retrieval_f1(predictions, references)
        assert result["retrieval_f1"] == 0.0
    
    def test_f1_harmonic_mean(self):
        """Test F1 is harmonic mean of precision and recall."""
        # Predicted [0, 1, 2], gold [0, 1]
        # precision = 2/3, recall = 1.0
        # F1 = 2 * (2/3 * 1.0) / (2/3 + 1.0) = 0.8
        predictions = [[0, 1, 2]]
        references = [[0, 1]]
        
        result = compute_retrieval_f1(predictions, references)
        assert abs(result["retrieval_f1"] - 0.8) < 0.01
    
    def test_both_empty_zero_f1(self):
        """Test F1 when both pred and gold are empty."""
        predictions = [[]]
        references = [[]]
        
        # precision = 0 (no predictions), recall = 1 (nothing to find)
        # F1 = 0 (division by zero handled)
        result = compute_retrieval_f1(predictions, references)
        # With our implementation: precision=0, recall=1, so F1=0
        assert result["retrieval_f1"] == 0.0

    def test_batch_f1_averages_per_example_scores(self):
        predictions = [[0], [0, 1]]
        references = [[0, 1], [0]]
        # Each example has F1=2/3; combining mean precision/recall gives 0.75.
        assert compute_retrieval_f1(predictions, references)["retrieval_f1"] == pytest.approx(2 / 3)

    def test_empty_prediction_does_not_borrow_recall_from_another_example(self):
        assert compute_retrieval_f1([[0], []], [[0], []]) == {"retrieval_f1": 0.5}

    @pytest.mark.parametrize("predictions,references", [([], []), ([None], [[0]])])
    def test_empty_batch_or_missing_prediction(self, predictions, references):
        assert compute_retrieval_f1(predictions, references) == {"retrieval_f1": 0.0}

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            compute_retrieval_f1([[0]], [])


class TestComputeAllRetrievalMetrics:
    """Tests for compute_all_retrieval_metrics."""
    
    def test_returns_all_metrics(self):
        """Test that all three metrics are returned."""
        predictions = [[0, 1]]
        references = [[0, 1]]
        
        result = compute_all_retrieval_metrics(predictions, references)
        
        assert "retrieval_precision" in result
        assert "retrieval_recall" in result
        assert "retrieval_f1" in result

    def test_batch_f1_matches_per_example_average(self):
        result = compute_all_retrieval_metrics([[0], [0, 1]], [[0, 1], [0]])
        assert result["retrieval_precision"] == 0.75
        assert result["retrieval_recall"] == 0.75
        assert result["retrieval_f1"] == pytest.approx(2 / 3)


class TestCombinedScore:
    """Tests for compute_combined_score."""
    
    def test_default_weights(self):
        """Test combined score with default weights (0.7/0.3)."""
        result = compute_combined_score(answer_f1=1.0, retrieval_f1=1.0)
        
        assert result["combined_score"] == 1.0
    
    def test_weighted_combination(self):
        """Test combined score calculation."""
        # 0.7 * 0.8 + 0.3 * 0.6 = 0.56 + 0.18 = 0.74
        result = compute_combined_score(answer_f1=0.8, retrieval_f1=0.6)
        
        assert abs(result["combined_score"] - 0.74) < 0.001
    
    def test_custom_weights(self):
        """Test combined score with custom weights."""
        # 0.5 * 0.8 + 0.5 * 0.6 = 0.40 + 0.30 = 0.70
        result = compute_combined_score(
            answer_f1=0.8,
            retrieval_f1=0.6,
            answer_weight=0.5,
            retrieval_weight=0.5,
        )
        
        assert abs(result["combined_score"] - 0.70) < 0.001
    
    def test_zero_scores(self):
        """Test combined score with zero inputs."""
        result = compute_combined_score(answer_f1=0.0, retrieval_f1=0.0)
        
        assert result["combined_score"] == 0.0


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_mismatched_lengths_raises(self):
        """Test that mismatched prediction/reference lengths raise error."""
        predictions = [[0, 1], [2, 3]]
        references = [[0, 1]]
        
        with pytest.raises(ValueError):
            compute_retrieval_precision(predictions, references)
    
    def test_empty_lists(self):
        """Test handling of empty prediction/reference lists."""
        predictions = []
        references = []
        
        result = compute_retrieval_precision(predictions, references)
        assert result["retrieval_precision"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
