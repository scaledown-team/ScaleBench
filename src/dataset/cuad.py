"""CUAD dataset loader.

CUAD (Contract Understanding Atticus Dataset) is a legal-contract span-extraction
QA benchmark — the "Extract" task on the existing ScaleDown benchmark pages
(Grok/OpenAI/Gemini) was run against CUAD with the F1 metric.

Loaded from chenghao/cuad_qa (a SQuAD-format mirror of the original CUAD
release; the official theatticusproject/cuad-qa loader script is no longer
loadable under current `datasets` versions, which dropped script-based
loaders — this mirror has the same fields as a plain Parquet dataset).
"""
import random
from typing import Iterator, Optional

from datasets import load_dataset as hf_load_dataset

from dataset.base import Dataset, Example


class CuadDataset(Dataset):
    """CUAD contract clause extraction dataset (SQuAD-format span QA)."""

    def __init__(self, name: str = "cuad"):
        super().__init__(name)

    def load_examples(
        self,
        split: str = "test",
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> Iterator[Example]:
        """Load and yield CUAD examples.

        Args:
            split: Dataset split (train/test).
            limit: Maximum number of examples to load (None = all).
            seed: Random seed for deterministic sampling.

        Yields:
            Example dicts with {id, context (contract text), question (clause
            prompt), answer (list of gold answer strings — empty list if the
            clause type isn't present in this contract)}.
        """
        hf_dataset = hf_load_dataset("chenghao/cuad_qa", split=split)
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
            answer_texts = row["answers"]["text"] if row["answers"]["text"] else [""]
            yield Example(
                id=row["id"],
                context=row["context"],
                question=row["question"],
                answer=answer_texts,
            )
