"""HotpotQA dataset loader."""
import random
from typing import Iterator, Optional
import datasets as hf_datasets
from dataset.base import Dataset, Example


class HotpotQADataset(Dataset):
    """HotpotQA multi-hop question answering dataset.
    
    Supports both 'distractor' and 'fullwiki' subsets.
    """

    def __init__(self, name: str = "hotpotqa", subset: str = "distractor"):
        """Initialize HotpotQA dataset.
        
        Args:
            name: Dataset name.
            subset: Subset to load ('distractor' or 'fullwiki').
        """
        super().__init__(name)
        self.subset = subset

    def load_examples(
        self,
        split: str = "validation",
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> Iterator[Example]:
        """Load and yield HotpotQA examples.
        
        Args:
            split: Dataset split (train/validation/test).
            limit: Maximum number of examples to load (None = all).
            seed: Random seed for deterministic sampling.
            
        Yields:
            Example dictionaries with {id, context, question, answer}.
        """
        # Load dataset from HuggingFace
        hf_dataset = hf_datasets.load_dataset("hotpotqa/hotpot_qa", self.subset, split=split)
        
        # Get indices for sampling
        total_size = len(hf_dataset)
        if limit is not None and limit < total_size:
            # Deterministic random sampling
            indices = random.Random(seed).sample(range(total_size), limit)
            indices.sort()  # Keep order for consistency
        else:
            indices = range(total_size)
        
        # Yield examples
        for idx in indices:
            raw_example = hf_dataset[idx]
            
            # Flatten context from sentences
            context_parts = []
            for title, sentences in zip(
                raw_example["context"]["title"],
                raw_example["context"]["sentences"]
            ):
                # Join sentences for each passage
                passage = " ".join(sentences)
                context_parts.append(f"Title: {title}\n{passage}")
            
            context = "\n\n".join(context_parts)
            
            # Create standardized example
            example = Example(
                id=raw_example["id"],
                context=context,
                question=raw_example["question"],
                answer=[raw_example["answer"]],  # Always return as list
            )
            
            yield example
