"""CLINC OOS dataset loader.

CLINC OOS (Larson et al., 2019) is an intent classification benchmark with
150 in-scope intents across 10 domains plus an out-of-scope class. Used here
for the "Classify" task on the ScaleDown benchmark pages.

Loaded from HuggingFace clinc_oos ("plus" config, which includes out-of-scope
examples in addition to the 150 in-domain intents).
"""
import random
from typing import Iterator, Optional

import datasets as hf_datasets

from dataset.base import Dataset, Example


class ClincOosDataset(Dataset):
    """CLINC OOS single-turn intent classification dataset."""

    def __init__(self, name: str = "clinc_oos"):
        super().__init__(name)

    def load_examples(
        self,
        split: str = "test",
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> Iterator[Example]:
        """Load and yield CLINC OOS examples.

        Args:
            split: Dataset split (train/validation/test).
            limit: Maximum number of examples to load (None = all).
            seed: Random seed for deterministic sampling.

        Yields:
            Example dicts with {id, context ("", unused), question (query text),
            answer ([intent label]), labels (full label list, for prompt building)}.
        """
        hf_dataset = hf_datasets.load_dataset("clinc_oos", "plus", split=split)
        label_names = hf_dataset.features["intent"].names

        all_examples = list(hf_dataset)
        total_size = len(all_examples)
        if limit is not None and limit < total_size:
            random.seed(seed)
            indices = random.sample(range(total_size), limit)
            indices.sort()
        else:
            indices = range(total_size)

        for idx in indices:
            row = all_examples[idx]
            yield Example(
                id=str(idx),
                context="",
                question=row["text"],
                answer=[label_names[row["intent"]]],
                labels=label_names,
            )
