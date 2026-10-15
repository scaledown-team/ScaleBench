"""Set-based retrieval metrics for passage retrieval evaluation.

These metrics evaluate how well a model identifies relevant passages from a
collection, using set-based precision, recall, and F1 calculations.

All functions follow the same interface:
    - predictions: List of predicted passage index lists
    - references: List of gold passage index lists
    - Returns: Dict with metric name and value
"""
from typing import List, Optional, Set, Callable


def _to_set(items: Optional[List[int]]) -> Set[int]:
    """Convert list to set, handling None values.
    
    Args:
        items: List of integers or None.
        
    Returns:
        Set of integers (empty set if None).
    """
    if items is None:
        return set()
    return set(items)


def compute_retrieval_precision(
    predictions: List[Optional[List[int]]],
    references: List[Optional[List[int]]],
    normalize_fn: Optional[Callable] = None,
) -> dict:
    """Compute set-based precision for passage retrieval.
    
    Precision = |predicted ∩ gold| / |predicted|
    
    If prediction is empty, precision is 0 for that example.
    
    Args:
        predictions: List of predicted passage index lists.
        references: List of gold passage index lists.
        normalize_fn: Unused, kept for interface consistency.
        
    Returns:
        Dict with 'retrieval_precision' key and float value.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions ({len(predictions)}) and references ({len(references)}) "
            "must have the same length"
        )
    
    if not predictions:
        return {"retrieval_precision": 0.0}
    
    precisions = []
    for pred, gold in zip(predictions, references):
        pred_set = _to_set(pred)
        gold_set = _to_set(gold)
        
        if not pred_set:
            # No predictions: precision is 0
            precisions.append(0.0)
        else:
            intersection = pred_set & gold_set
            precisions.append(len(intersection) / len(pred_set))
    
    return {"retrieval_precision": sum(precisions) / len(precisions)}


def compute_retrieval_recall(
    predictions: List[Optional[List[int]]],
    references: List[Optional[List[int]]],
    normalize_fn: Optional[Callable] = None,
) -> dict:
    """Compute set-based recall for passage retrieval.
    
    Recall = |predicted ∩ gold| / |gold|
    
    If gold is empty, recall is 1.0 for that example (nothing to find).
    
    Args:
        predictions: List of predicted passage index lists.
        references: List of gold passage index lists.
        normalize_fn: Unused, kept for interface consistency.
        
    Returns:
        Dict with 'retrieval_recall' key and float value.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions ({len(predictions)}) and references ({len(references)}) "
            "must have the same length"
        )
    
    if not predictions:
        return {"retrieval_recall": 0.0}
    
    recalls = []
    for pred, gold in zip(predictions, references):
        pred_set = _to_set(pred)
        gold_set = _to_set(gold)
        
        if not gold_set:
            # No gold passages: recall is 1.0 (nothing to find)
            recalls.append(1.0)
        else:
            intersection = pred_set & gold_set
            recalls.append(len(intersection) / len(gold_set))
    
    return {"retrieval_recall": sum(recalls) / len(recalls)}


def compute_retrieval_f1(
    predictions: List[Optional[List[int]]],
    references: List[Optional[List[int]]],
    normalize_fn: Optional[Callable] = None,
) -> dict:
    """Compute macro-averaged set-based F1 for passage retrieval.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Compute F1 per example, then average over the batch.
    F1 is 0 if both precision and recall are 0.
    
    Args:
        predictions: List of predicted passage index lists.
        references: List of gold passage index lists.
        normalize_fn: Unused, kept for interface consistency.
        
    Returns:
        Dict with 'retrieval_f1' key and float value.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions ({len(predictions)}) and references ({len(references)}) "
            "must have the same length"
        )

    scores = []
    for pred, gold in zip(predictions, references):
        pred_set = _to_set(pred)
        gold_set = _to_set(gold)
        total = len(pred_set) + len(gold_set)
        scores.append(2 * len(pred_set & gold_set) / total if total else 0.0)

    return {"retrieval_f1": sum(scores) / len(scores) if scores else 0.0}


def compute_all_retrieval_metrics(
    predictions: List[Optional[List[int]]],
    references: List[Optional[List[int]]],
    normalize_fn: Optional[Callable] = None,
) -> dict:
    """Compute all retrieval metrics in a single call.
    
    Args:
        predictions: List of predicted passage index lists.
        references: List of gold passage index lists.
        normalize_fn: Unused, kept for interface consistency.
        
    Returns:
        Dict with 'retrieval_precision', 'retrieval_recall', and 'retrieval_f1' keys.
    """
    result = {}
    result.update(compute_retrieval_precision(predictions, references))
    result.update(compute_retrieval_recall(predictions, references))
    result.update(compute_retrieval_f1(predictions, references))
    return result


def compute_combined_score(
    answer_f1: float,
    retrieval_f1: float,
    answer_weight: float = 0.7,
    retrieval_weight: float = 0.3,
) -> dict:
    """Compute weighted combined score from answer and retrieval F1.
    
    Combined = answer_weight * answer_f1 + retrieval_weight * retrieval_f1
    
    Args:
        answer_f1: F1 score for answer quality (0-1).
        retrieval_f1: F1 score for passage retrieval (0-1).
        answer_weight: Weight for answer F1 (default 0.7).
        retrieval_weight: Weight for retrieval F1 (default 0.3).
        
    Returns:
        Dict with 'combined_score' key and float value.
    """
    combined = answer_weight * answer_f1 + retrieval_weight * retrieval_f1
    return {"combined_score": combined}
