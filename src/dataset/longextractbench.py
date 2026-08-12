"""LongExtractBench dataset loader.

LongExtractBench-50 (micro1-inc/longextract-bench-50 on HuggingFace) is a
50-document benchmark for schema-driven structured extraction from long,
table-heavy PDFs (financial filings, government statistics, healthcare
reports, etc). Each example is a folder containing:
    document.pdf     - the source document
    schema.json       - target JSON Schema for extraction
    ground_truth.json - human-reconciled gold extraction

Unlike the other datasets in this repo, LongExtractBench is not a
`datasets`-loadable parquet dataset (it's a raw file repo, one folder per
example), so this loader uses `huggingface_hub.snapshot_download` to pull
the whole repo locally and reads each folder directly. PDF text extraction
mirrors dataset/financebench.py's approach (PyMuPDF).
"""
import json
import random
from pathlib import Path
from typing import Iterator, Optional

from huggingface_hub import snapshot_download

from dataset.base import Dataset, Example

REPO_ID = "micro1-inc/longextract-bench-50"


def _load_pdf_text(pdf_path: Path) -> str:
    """Extract full text from a PDF using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Full document text with pages joined by double newlines.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages)


class LongExtractBenchDataset(Dataset):
    """LongExtractBench-50 schema-driven structured extraction dataset."""

    def __init__(self, name: str = "longextractbench", cache_dir: Optional[str] = None):
        """Initialize LongExtractBench dataset.

        Args:
            name: Dataset name.
            cache_dir: Optional huggingface_hub cache directory override.
        """
        super().__init__(name)
        self.cache_dir = cache_dir

    def load_examples(
        self,
        split: str = "test",
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> Iterator[Example]:
        """Load and yield LongExtractBench examples.

        Args:
            split: Unused — LongExtractBench-50 has a single flat set of examples.
            limit: Maximum number of examples to load (None = all 50).
            seed: Random seed for deterministic sampling.

        Yields:
            Example dicts with {id, context (document text), question (schema as
            JSON string, for logging), answer ([ground truth as JSON string]),
            schema (parsed dict), ground_truth (parsed dict)}.
        """
        repo_dir = Path(snapshot_download(
            repo_id=REPO_ID, repo_type="dataset", cache_dir=self.cache_dir,
        ))

        example_dirs = sorted(
            d for d in repo_dir.iterdir()
            if d.is_dir() and (d / "document.pdf").exists()
            and (d / "schema.json").exists()
            and (d / "ground_truth.json").exists()
        )

        total_size = len(example_dirs)
        if limit is not None and limit < total_size:
            random.seed(seed)
            indices = random.sample(range(total_size), limit)
            indices.sort()
        else:
            indices = range(total_size)

        for idx in indices:
            d = example_dirs[idx]
            schema = json.loads((d / "schema.json").read_text())
            ground_truth = json.loads((d / "ground_truth.json").read_text())
            context = _load_pdf_text(d / "document.pdf")

            yield Example(
                id=d.name,
                context=context,
                question=json.dumps(schema),
                answer=[json.dumps(ground_truth)],
                schema=schema,
                ground_truth=ground_truth,
            )
