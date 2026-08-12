"""Field-level extraction metrics for schema-driven structured extraction (e.g. LongExtractBench).

Scores a predicted JSON object against a gold JSON object by flattening both
into dotted-path -> stringified-value pairs and computing set-based
precision/recall/F1 over those pairs. This is a first-pass scoring method:
it does exact string match per field after light normalization (strip,
lowercase, collapse whitespace) — no fuzzy numeric tolerance, no date
canonicalization, no schema validation. Good enough to get a directional
signal; revisit before publishing a precise number.

Follows the one-metric-name -> one-function -> one-key convention used
elsewhere in this repo (see metrics/retrieval.py), so each of precision/
recall/F1/parse-rate is its own registrable metric rather than one function
returning several keys.
"""
import json
from typing import List, Dict, Callable, Optional, Any


def _flatten(obj: Any, prefix: str = "") -> Dict[str, str]:
    """Flatten a nested JSON-like structure into {dotted.path: stringified_value}.

    Args:
        obj: Parsed JSON value (dict, list, or scalar).
        prefix: Dotted-path prefix accumulated from parent keys.

    Returns:
        Flat dict mapping dotted paths to normalized string values.
    """
    out: Dict[str, str] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = str(obj).strip().lower()
    return out


def _safe_parse(text: str) -> Optional[dict]:
    """Parse a JSON object from text, tolerant of stray whitespace. Returns None on failure."""
    try:
        parsed = json.loads(text.strip())
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, AttributeError):
        return None


def _per_example_scores(predictions: List[str], references: List[List[str]]) -> List[Dict[str, float]]:
    """Shared per-example precision/recall/F1/parsed-ok computation."""
    scores = []
    for pred_text, refs in zip(predictions, references):
        gold_text = refs[0] if refs else "{}"
        gold = _flatten(_safe_parse(gold_text) or {})
        pred_parsed = _safe_parse(pred_text)
        pred = _flatten(pred_parsed or {})

        if not gold and not pred:
            scores.append({"precision": 1.0, "recall": 1.0, "f1": 1.0, "parsed_ok": 1.0 if pred_parsed is not None else 0.0})
            continue
        if not pred or not gold:
            scores.append({"precision": 0.0, "recall": 0.0, "f1": 0.0, "parsed_ok": 1.0 if pred_parsed is not None else 0.0})
            continue

        correct = sum(1 for k, v in pred.items() if k in gold and gold[k] == v)
        precision = correct / len(pred)
        recall = correct / len(gold)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        scores.append({"precision": precision, "recall": recall, "f1": f1, "parsed_ok": 1.0 if pred_parsed is not None else 0.0})
    return scores


def compute_field_precision(
    predictions: List[str], references: List[List[str]], normalize_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, float]:
    scores = _per_example_scores(predictions, references)
    n = len(scores) or 1
    return {"field_precision": sum(s["precision"] for s in scores) / n}


def compute_field_recall(
    predictions: List[str], references: List[List[str]], normalize_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, float]:
    scores = _per_example_scores(predictions, references)
    n = len(scores) or 1
    return {"field_recall": sum(s["recall"] for s in scores) / n}


def compute_field_f1(
    predictions: List[str], references: List[List[str]], normalize_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, float]:
    scores = _per_example_scores(predictions, references)
    n = len(scores) or 1
    return {"field_f1": sum(s["f1"] for s in scores) / n}


def compute_parse_rate(
    predictions: List[str], references: List[List[str]], normalize_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, float]:
    scores = _per_example_scores(predictions, references)
    n = len(scores) or 1
    return {"parse_rate": sum(s["parsed_ok"] for s in scores) / n}
