"""ETHIC dataset loader."""
import random
from typing import Iterator, Optional
import datasets as hf_datasets
from dataset.base import Dataset, Example


VALID_SUBSETS = ["Attributing", "Recalling", "Organizing"]


class ETHICDataset(Dataset):
    """ETHIC long-context evaluation dataset.
    
    Supports subsets: Attributing, Recalling, Organizing.
    Summarizing subset is excluded as it lacks gold answers.
    """

    def __init__(self, name: str = "ethic", subset: str = "Recalling"):
        """Initialize ETHIC dataset.
        
        Args:
            name: Dataset name.
            subset: Subset to load (must be one of VALID_SUBSETS).
            
        Raises:
            ValueError: If subset is not in VALID_SUBSETS.
        """
        super().__init__(name)
        if subset not in VALID_SUBSETS:
            raise ValueError(
                f"Invalid subset '{subset}'. Must be one of: {VALID_SUBSETS}"
            )
        self.subset = subset

    def load_examples(
        self,
        split: str = "test",
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> Iterator[Example]:
        """Load and yield ETHIC examples.
        
        Args:
            split: Dataset split (ETHIC only has 'test').
            limit: Maximum number of examples to load (None = all).
            seed: Random seed for deterministic sampling.
            
        Yields:
            Example dictionaries with {id, context, question, answer} plus metadata.
        """
        # Load dataset from HuggingFace
        hf_dataset = hf_datasets.load_dataset(
            "dmis-lab/ETHIC", self.subset, split=split
        )
        
        # Get indices for sampling
        total_size = len(hf_dataset)
        if limit is not None and limit < total_size:
            indices = random.Random(seed).sample(range(total_size), limit)
            indices.sort()
        else:
            indices = range(total_size)
        
        # Yield examples
        for idx in indices:
            raw_example = hf_dataset[idx]
            
            # Create standardized example
            # User_msg contains the full prompt with embedded context
            # question is empty since User_msg has everything
            example = Example(
                id=raw_example["ID"],
                context=raw_example["User_msg"],
                question="",
                answer=raw_example["Answer"],
                # Additional metadata
                domain=raw_example["Domain"],
                ethic_task=raw_example["Task"],
                system_msg=raw_example["System_msg"],
            )
            
            yield example
