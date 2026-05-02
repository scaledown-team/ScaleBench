"""SQuAD v2 dataset loader.

SQuAD v2 (Stanford Question Answering Dataset) includes both answerable
questions and unanswerable questions where the answer is not in the context.
"""
import random
from typing import Iterator, Optional
import datasets as hf_datasets
from dataset.base import Dataset, Example


class SQuADv2Dataset(Dataset):
    """SQuAD v2 question answering dataset."""

    def __init__(self, name: str = "squad_v2"):
        """Initialize SQuAD v2 dataset.
        
        Args:
            name: Dataset name.
        """
        super().__init__(name)

    def load_examples(
        self,
        split: str = "validation",
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> Iterator[Example]:
        """Load and yield SQuAD v2 examples.
        
        Args:
            split: Dataset split (train/validation).
            limit: Maximum number of examples to load (None = all).
            seed: Random seed for deterministic sampling.
            
        Yields:
            Example dictionaries with {id, context, question, answer}.
        """
        # Load dataset from HuggingFace
        hf_dataset = hf_datasets.load_dataset("rajpurkar/squad_v2", split=split)
        
        # Get indices for sampling
        total_size = len(hf_dataset)
        if limit is not None and limit < total_size:
            # Deterministic random sampling
            random.seed(seed)
            indices = random.sample(range(total_size), limit)
            indices.sort()  # Keep order for consistency
        else:
            indices = range(total_size)
        
        # Yield examples
        for idx in indices:
            raw_example = hf_dataset[idx]
            
            # SQuAD v2 answers are in a list under the 'answers' key
            # Each answer has 'text' and 'answer_start'
            # If the question is unanswerable, 'text' is an empty list
            answers = raw_example["answers"]["text"]
            
            # Create standardized example
            example = Example(
                id=raw_example["id"],
                context=raw_example["context"],
                question=raw_example["question"],
                answer=list(answers),  # Ensure it's a list (already is, but for safety)
            )
            
            yield example
