"""MS-MARCO dataset loader."""
import random
from typing import Iterator, Optional, List
import datasets as hf_datasets
from dataset.base import Dataset, Example


class MSMARCODataset(Dataset):
    """MS-MARCO v2.1 passage ranking and QA dataset.
    
    Loads examples with passages, answers, and retrieval labels.
    Filters out examples without valid answers or retrieval ground truth.
    """

    def __init__(self, name: str = "msmarco"):
        """Initialize MS-MARCO dataset.
        
        Args:
            name: Dataset name.
        """
        super().__init__(name)

    def _has_valid_answer(self, answers: List[str]) -> bool:
        """Check if example has a valid answer.
        
        Filters out examples where answers only contain "No Answer Present."
        
        Args:
            answers: List of answer strings.
            
        Returns:
            True if example has at least one valid answer.
        """
        if not answers:
            return False
        for answer in answers:
            if answer.strip() and answer.strip() != "No Answer Present.":
                return True
        return False

    def _has_retrieval_ground_truth(self, is_selected: List[int]) -> bool:
        """Check if example has retrieval ground truth.
        
        Args:
            is_selected: List of 0/1 indicators for each passage.
            
        Returns:
            True if at least one passage is marked as selected.
        """
        return any(s == 1 for s in is_selected)

    def _build_indexed_context(self, passage_texts: List[str]) -> str:
        """Build context string with indexed passages.
        
        Format:
            [Passage 0]
            <passage_text_0>
            
            [Passage 1]
            <passage_text_1>
            ...
        
        Args:
            passage_texts: List of passage text strings.
            
        Returns:
            Formatted context string with passage indices.
        """
        parts = []
        for idx, text in enumerate(passage_texts):
            parts.append(f"[Passage {idx}]\n{text}")
        return "\n\n".join(parts)

    def load_examples(
        self,
        split: str = "validation",
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> Iterator[Example]:
        """Load and yield MS-MARCO examples.
        
        Args:
            split: Dataset split (train/validation/test).
            limit: Maximum number of examples to load (None = all).
            seed: Random seed for deterministic sampling.
            
        Yields:
            Example dictionaries with {id, context, question, answer} plus
            retrieval-specific fields (well_formed_answers, passage_labels,
            selected_indices).
        """
        # Load dataset from HuggingFace
        hf_dataset = hf_datasets.load_dataset(
            "microsoft/ms_marco", "v2.1", split=split
        )
        
        # Filter to valid examples (has answer and retrieval ground truth)
        valid_indices = []
        for idx in range(len(hf_dataset)):
            raw = hf_dataset[idx]
            answers = raw["answers"]
            is_selected = raw["passages"]["is_selected"]
            
            if self._has_valid_answer(answers) and self._has_retrieval_ground_truth(is_selected):
                valid_indices.append(idx)
        
        # Apply limit with deterministic sampling
        if limit is not None and limit < len(valid_indices):
            valid_indices = random.Random(seed).sample(valid_indices, limit)
            valid_indices.sort()
        
        # Yield examples
        for idx in valid_indices:
            raw_example = hf_dataset[idx]
            
            passages = raw_example["passages"]
            passage_texts = passages["passage_text"]
            is_selected = passages["is_selected"]
            
            # Build indexed context
            context = self._build_indexed_context(passage_texts)
            
            # Get selected indices for retrieval ground truth
            selected_indices = [i for i, s in enumerate(is_selected) if s == 1]
            
            # Filter valid answers (exclude "No Answer Present.")
            valid_answers = [
                a for a in raw_example["answers"]
                if a.strip() and a.strip() != "No Answer Present."
            ]
            
            # Get well-formed answers (may be empty)
            well_formed = raw_example.get("wellFormedAnswers", [])
            if well_formed is None:
                well_formed = []
            # Filter empty strings from well_formed answers
            well_formed = [w for w in well_formed if w and w.strip()]
            
            # Create standardized example
            example = Example(
                id=str(raw_example["query_id"]),
                context=context,
                question=raw_example["query"],
                answer=valid_answers,
                # MS-MARCO specific fields
                well_formed_answers=well_formed,
                passage_labels=is_selected,
                selected_indices=selected_indices,
                query_type=raw_example.get("query_type", ""),
            )
            
            yield example
