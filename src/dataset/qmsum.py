"""QMSum dataset loader.

QMSum is a query-based meeting summarization benchmark (NAACL 2021).
Given a meeting transcript and a query, the model must generate an abstractive
summary answering the query.

Loaded from HuggingFace Ahren09/QMSum (parquet-based, all splits labeled).

Each row in the HF dataset is one meeting document containing multiple queries
(general_query_list and specific_query_list). This loader explodes each meeting
into one example per query-answer pair.

Splits: train (162 meetings), validation (35 meetings), test (35 meetings).
All splits have reference answers.
"""
import random
from typing import Iterator, Optional, List, Dict, Any

import datasets as hf_datasets

from dataset.base import Dataset, Example


def _format_transcript(meeting_transcripts: List[Dict[str, Any]]) -> str:
    """Format speaker-turn list into a single transcript string.

    Args:
        meeting_transcripts: List of {"speaker": str, "content": str} dicts.

    Returns:
        Formatted transcript string.
    """
    return "\n".join(
        f"{turn['speaker']}: {turn['content']}"
        for turn in meeting_transcripts
    )


class QMSumDataset(Dataset):
    """QMSum query-based meeting summarization dataset.

    Loads from HuggingFace Ahren09/QMSum and explodes each meeting document
    into individual query-answer examples (both general and specific queries).
    """

    def __init__(self, name: str = "qmsum"):
        super().__init__(name)

    def load_examples(
        self,
        split: str = "test",
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> Iterator[Example]:
        """Load and yield QMSum examples (one per query-answer pair).

        Args:
            split: Dataset split (train/validation/test). All have reference answers.
            limit: Maximum number of examples to load (None = all).
            seed: Random seed for deterministic sampling when limit is set.

        Yields:
            Example dicts with {id, context (transcript), question (query), answer}.
        """
        hf_dataset = hf_datasets.load_dataset("Ahren09/QMSum", split=split)

        # Explode meetings into individual query-answer pairs
        all_examples = []
        for row in hf_dataset:
            transcript = _format_transcript(row["meeting_transcripts"])
            meeting_id = row["id"]

            queries = (
                list(row.get("general_query_list") or [])
                + list(row.get("specific_query_list") or [])
            )

            for i, q in enumerate(queries):
                query = q.get("query", "").strip()
                answer = q.get("answer", "").strip()
                if not query or not answer:
                    continue
                all_examples.append(Example(
                    id=f"{meeting_id}-q{i}",
                    context=transcript,
                    question=query,
                    answer=[answer],
                ))

        if limit is not None and limit < len(all_examples):
            all_examples = random.Random(seed).sample(all_examples, limit)
            all_examples.sort(key=lambda e: e["id"])

        yield from all_examples
