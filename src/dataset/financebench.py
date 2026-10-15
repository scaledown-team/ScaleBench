"""FinanceBench dataset loader."""
import random
from pathlib import Path
from typing import Iterator, Optional

from datasets import load_dataset as hf_load_dataset

from dataset.base import Dataset, Example

# Default path to cloned patronus-ai/financebench PDF directory
DEFAULT_PDF_DIR = Path(
    "/Users/soham/Desktop/scaledown-project/financebench-data/pdfs"
)


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


class FinanceBenchDataset(Dataset):
    """FinanceBench financial question answering dataset.

    150 open-source annotated examples from SEC filings
    (10-K, 10-Q, 8-K). Uses full PDF document text as
    context for long-context evaluation (~100k tokens).

    Source: PatronusAI/financebench on HuggingFace (CC-BY-NC-4.0).
    PDFs: https://github.com/patronus-ai/financebench
    """

    def __init__(
        self,
        name: str = "financebench",
        pdf_dir: Optional[Path] = None,
    ):
        """Initialize FinanceBench dataset.

        Args:
            name: Dataset name.
            pdf_dir: Path to directory containing {doc_name}.pdf files.
                     Defaults to DEFAULT_PDF_DIR. If None or directory
                     does not exist, falls back to evidence_text_full_page.
        """
        super().__init__(name)
        self.pdf_dir = Path(pdf_dir) if pdf_dir else DEFAULT_PDF_DIR

    def load_examples(
        self,
        split: str = "train",
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> Iterator[Example]:
        """Load and yield FinanceBench examples.

        Context is the full SEC filing PDF text when available,
        falling back to evidence_text_full_page otherwise.

        Args:
            split: Dataset split. FinanceBench only has 'train'.
            limit: Maximum number of examples to load (None = all).
            seed: Random seed for deterministic sampling.

        Yields:
            Example dicts with {id, context, question, answer}.
        """
        use_pdfs = self.pdf_dir.is_dir()
        if not use_pdfs:
            import logging
            logging.getLogger(__name__).warning(
                f"PDF directory not found: {self.pdf_dir}. "
                "Falling back to evidence_text_full_page."
            )

        raw_dataset = hf_load_dataset(
            "PatronusAI/financebench", split="train"
        )
        all_examples = list(raw_dataset)

        # Deterministic sampling
        total_size = len(all_examples)
        if limit is not None and limit < total_size:
            indices = random.Random(seed).sample(range(total_size), limit)
            indices.sort()
        else:
            indices = range(total_size)

        # Cache PDF text per doc_name to avoid re-reading the same PDF
        pdf_cache: dict = {}

        for idx in indices:
            raw = all_examples[idx]
            doc_name = raw.get("doc_name", "")
            context_source = "fallback"

            if use_pdfs and doc_name:
                if doc_name not in pdf_cache:
                    pdf_path = self.pdf_dir / f"{doc_name}.pdf"
                    if pdf_path.exists():
                        pdf_cache[doc_name] = _load_pdf_text(pdf_path)
                    else:
                        pdf_cache[doc_name] = None
                context = pdf_cache[doc_name]
                if context:
                    context_source = "pdf"

            if not use_pdfs or not context:
                # Fallback: concatenate evidence full-page text
                evidence = raw.get("evidence") or []
                full_pages = [
                    e.get("evidence_text_full_page", "")
                    for e in evidence
                    if e.get("evidence_text_full_page")
                ]
                context = "\n\n".join(full_pages) if full_pages else ""
                context_source = "fallback"

            # Answer is a single string; wrap in list for compatibility
            answer_str = raw.get("answer", "")
            answer = [answer_str] if answer_str else []

            yield Example(
                id=raw.get("financebench_id", str(idx)),
                context=context,
                question=raw.get("question", ""),
                answer=answer,
                context_source=context_source,
            )
