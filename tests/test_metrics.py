"""Unit tests for metrics."""
import pytest
from metrics.classification import compute_exact_match, compute_f1


def test_exact_match_perfect():
    """Test exact_match with perfect matches."""
    predictions = ["paris", "london", "tokyo"]
    references = [["paris"], ["london"], ["tokyo"]]
    
    result = compute_exact_match(predictions, references)
    assert result["exact_match"] == 1.0


def test_exact_match_partial():
    """Test exact_match with some matches."""
    predictions = ["paris", "berlin", "tokyo"]
    references = [["paris"], ["london"], ["tokyo"]]
    
    result = compute_exact_match(predictions, references)
    assert abs(result["exact_match"] - 0.6667) < 0.01


def test_exact_match_with_normalization():
    """Test exact_match with normalization function."""
    predictions = ["Paris", "LONDON"]
    references = [["paris"], ["london"]]
    
    normalize_fn = lambda x: x.lower()
    result = compute_exact_match(predictions, references, normalize_fn=normalize_fn)
    assert result["exact_match"] == 1.0


def test_f1_perfect():
    """Test F1 with perfect matches."""
    predictions = ["paris france", "london england"]
    references = [["paris france"], ["london england"]]
    
    result = compute_f1(predictions, references)
    assert result["f1"] == 1.0


def test_f1_partial_overlap():
    """Test F1 with partial token overlap."""
    predictions = ["paris"]
    references = [["paris france"]]
    
    result = compute_f1(predictions, references)
    # precision=1.0 (paris in paris), recall=0.5 (paris in paris france)
    # f1 = 2 * (1.0 * 0.5) / (1.0 + 0.5) = 0.6667
    assert abs(result["f1"] - 0.6667) < 0.01


def test_f1_no_overlap():
    """Test F1 with no overlap."""
    predictions = ["berlin"]
    references = [["paris"]]
    
    result = compute_f1(predictions, references)
    assert result["f1"] == 0.0


def test_f1_multi_reference():
    """Test F1 with multiple references (takes max)."""
    predictions = ["paris"]
    references = [["london", "paris"]]
    
    result = compute_f1(predictions, references)
    assert result["f1"] == 1.0  # Perfect match with second reference


def test_f1_with_normalization():
    """Test F1 with normalization function."""
    predictions = ["The Paris"]
    references = [["paris"]]
    
    normalize_fn = lambda x: x.lower().replace("the ", "")
    result = compute_f1(predictions, references, normalize_fn=normalize_fn)
    assert result["f1"] == 1.0


def test_exact_match_accepts_any_normalized_reference():
    assert compute_exact_match(
        ["PARIS"], [["London", "Paris"]], normalize_fn=str.lower
    ) == {"exact_match": 1.0}


def test_exact_match_reference_order_does_not_change_score():
    predictions = ["Paris", "Berlin"]
    references = [["London", "Paris"], ["Tokyo", "Rome"]]
    reversed_references = [list(reversed(refs)) for refs in references]
    assert compute_exact_match(predictions, references) == {"exact_match": 0.5}
    assert compute_exact_match(predictions, reversed_references) == {"exact_match": 0.5}


def test_exact_match_empty_batch():
    assert compute_exact_match([], []) == {"exact_match": 0.0}


def test_exact_match_empty_reference_list():
    assert compute_exact_match(["", "answer"], [[], []]) == {"exact_match": 0.5}


@pytest.mark.parametrize("metric", [compute_exact_match, compute_f1])
@pytest.mark.parametrize("predictions,references", [(["a"], []), ([], [["a"]])])
def test_qa_metrics_reject_mismatched_batch_lengths(metric, predictions, references):
    with pytest.raises(ValueError, match="same length"):
        metric(predictions, references)


@pytest.mark.parametrize(
    "prediction,reference,expected",
    [
        ("red red blue", "red red blue", 1.0),
        ("red red", "red red blue", 0.8),
        ("red red blue", "red red", 0.8),
        ("red red red", "red", 0.5),
    ],
)
def test_f1_counts_repeated_tokens(prediction, reference, expected):
    assert compute_f1([prediction], [[reference]])["f1"] == pytest.approx(expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
