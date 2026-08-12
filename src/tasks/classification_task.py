"""Intent classification task (e.g. CLINC OOS)."""
import re
from typing import Dict, Any, Optional

from models.base import GenerationInput
from tasks.base import Task

_SYSTEM_PROMPT_TEMPLATE = (
    "Classify the user's message into exactly one of the following intents:\n"
    "{labels}\n\n"
    "Respond with only the intent label, exactly as written above, and nothing else."
)


class ClassificationTask(Task):
    """Single-label intent classification task.

    Designed for CLINC OOS and similar closed-label classification benchmarks.
    Expects each example to carry a `labels` field (the full label set) so the
    prompt can enumerate valid intents.
    """

    def __init__(self, name: str = "classification", dataset_name: Optional[str] = None):
        super().__init__(name)
        self.dataset_name = dataset_name

    def build_prompt(self, example: Dict[str, Any], **kwargs) -> GenerationInput:
        """Build GenerationInput for intent classification.

        Args:
            example: Example with 'question' (utterance) and 'labels' (label list).
            **kwargs: Remaining generation parameters forwarded to GenerationInput.

        Returns:
            GenerationInput configured for the target client.
        """
        # See tasks/rag_task.py for why this pop is needed.
        kwargs.pop("model_mode", None)

        labels = example.get("labels", [])
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(labels=", ".join(labels))
        return GenerationInput(
            system_prompt=system_prompt,
            context=None,
            user_prompt=example.get("question", ""),
            **kwargs,
        )

    def postprocess(self, output: str) -> str:
        """Take the first non-empty line as the predicted label."""
        for line in output.strip().split("\n"):
            line = line.strip()
            if line:
                return line
        return output.strip()

    def normalize(self, text: str) -> str:
        """Normalize label text for comparison: lowercase, spaces/hyphens -> underscore."""
        text = text.strip().lower()
        text = re.sub(r"[\s\-]+", "_", text)
        text = re.sub(r"[^a-z0-9_]", "", text)
        return text
