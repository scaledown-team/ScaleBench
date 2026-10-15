"""LongBench dataset loader."""
import json
import random
from pathlib import Path
from typing import Iterator, Optional
from urllib.request import urlretrieve
import zipfile

from dataset.base import Dataset, Example


VALID_TASKS = ["narrativeqa", "qasper", "multifieldqa_en", "2wikimqa"]

# Base URL for LongBench data
LONGBENCH_DATA_URL = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"
LONGBENCH_CACHE_DIR = Path.home() / ".cache" / "longbench"


class LongBenchDataset(Dataset):
    """LongBench long-context QA dataset.
    
    Supports QA tasks: narrativeqa, qasper, multifieldqa_en, 2wikimqa.
    
    Note: Downloads data directly from HuggingFace as JSONL files to avoid
    deprecated dataset scripts.
    """

    def __init__(self, name: str = "longbench", task: str = "narrativeqa"):
        """Initialize LongBench dataset.
        
        Args:
            name: Dataset name.
            task: LongBench task name (must be one of VALID_TASKS).
            
        Raises:
            ValueError: If task is not in VALID_TASKS.
        """
        super().__init__(name)
        if task is not None and task not in VALID_TASKS:
            raise ValueError(f"Invalid task '{task}'. Must be one of: {VALID_TASKS}")
        self.task = task
        self._ensure_data_downloaded()

    def _ensure_data_downloaded(self):
        """Download LongBench data if not already cached."""
        data_dir = LONGBENCH_CACHE_DIR / "data"
        task_file = data_dir / f"{self.task}.jsonl"
        
        if task_file.exists():
            return  # Already downloaded
        
        # Create cache directory
        LONGBENCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Download zip file
        zip_path = LONGBENCH_CACHE_DIR / "data.zip"
        if not zip_path.exists():
            print(f"Downloading LongBench data from {LONGBENCH_DATA_URL}...")
            urlretrieve(LONGBENCH_DATA_URL, zip_path)
            print("Download complete.")
        
        # Extract zip file
        print(f"Extracting LongBench data to {LONGBENCH_CACHE_DIR}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(LONGBENCH_CACHE_DIR)
        print("Extraction complete.")

    def load_examples(
        self,
        split: str = "test",
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> Iterator[Example]:
        """Load and yield LongBench examples from JSONL files.
        
        Args:
            split: Dataset split (LongBench uses 'test', parameter kept for API compatibility).
            limit: Maximum number of examples to load (None = all).
            seed: Random seed for deterministic sampling.
            
        Yields:
            Example dictionaries with {id, context, question, answer}.
        """
        # Load data from JSONL file
        data_file = LONGBENCH_CACHE_DIR / "data" / f"{self.task}.jsonl"
        
        if not data_file.exists():
            raise FileNotFoundError(
                f"LongBench data file not found: {data_file}. "
                "This should have been downloaded automatically. "
                "Try deleting ~/.cache/longbench and running again."
            )
        
        # Read all examples from JSONL
        all_examples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                all_examples.append(json.loads(line))
        
        # Get indices for sampling
        total_size = len(all_examples)
        if limit is not None and limit < total_size:
            # Deterministic random sampling
            indices = random.Random(seed).sample(range(total_size), limit)
            indices.sort()
        else:
            indices = range(total_size)
        
        # Yield examples
        for idx in indices:
            raw_example = all_examples[idx]
            
            # Extract fields
            context = raw_example.get("context", "")
            question = raw_example.get("input", "")
            answers = raw_example.get("answers", [])
            
            # Ensure answers is a list
            if isinstance(answers, str):
                answers = [answers]
            
            # Create standardized example with task-prefixed ID
            example = Example(
                id=f"{self.task}_{idx}",
                context=context,
                question=question,
                answer=answers,
            )
            
            yield example

