"""Schema-driven structured extraction task (e.g. LongExtractBench)."""
import json
from typing import Dict, Any, Optional

from models.base import GenerationInput
from tasks.base import Task

_SYSTEM_PROMPT_TEMPLATE = (
    "Extract the requested fields from the document below and return ONLY a "
    "single valid JSON object matching this schema. Do not include commentary, "
    "explanations, or markdown code fences — output raw JSON only.\n\n"
    "Schema:\n{schema}"
)


class ExtractionTask(Task):
    """Structured extraction task scored via field-level F1 on parsed JSON.

    Designed for LongExtractBench and similar schema-driven extraction
    benchmarks. Expects each example to carry a `schema` field (a JSON Schema
    dict) so the prompt can describe the target structure.
    """

    def __init__(self, name: str = "extraction", dataset_name: Optional[str] = None):
        super().__init__(name)
        self.dataset_name = dataset_name

    def build_prompt(self, example: Dict[str, Any], **kwargs) -> GenerationInput:
        """Build GenerationInput for schema-driven extraction.

        Args:
            example: Example with 'context' (document text) and 'schema' (JSON Schema dict).
            **kwargs: Remaining generation parameters forwarded to GenerationInput.

        Returns:
            GenerationInput configured for the target client.
        """
        schema = example.get("schema", {})
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(schema=json.dumps(schema))
        return GenerationInput(
            system_prompt=system_prompt,
            context=f"Document:\n{example.get('context', '')}",
            user_prompt="Return the extracted JSON object now.",
            response_schema={"name": "extraction", "schema": schema},
            **kwargs,
        )

    def postprocess(self, output: str) -> str:
        """Strip markdown code fences if the model wrapped the JSON anyway."""
        text = output.strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        return text

    def normalize(self, text: str) -> str:
        """No-op — the field_f1 metric parses and compares JSON directly."""
        return text
