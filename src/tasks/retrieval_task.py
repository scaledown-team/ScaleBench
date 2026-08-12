"""Retrieval task for MS-MARCO with structured output (answer + passages)."""
from typing import Dict, Any, Optional, Union
from tasks.base import Task
from prompts.rag_prompts import RAG_PROMPTS, DEFAULT_RAG_PROMPT
from models.structured_output import MSMARCO_RESPONSE_SCHEMA, parse_msmarco_response


class RetrievalTask(Task):
    """Retrieval task that returns structured output with answer and passage indices.
    
    This task is designed for datasets like MS-MARCO where both answer quality
    and passage retrieval accuracy need to be evaluated together.
    
    Uses native structured output APIs (OpenAI response_format, Gemini response_schema)
    for guaranteed JSON output format compliance.
    
    Unlike other tasks that return a string from postprocess(), this task returns
    a dict with 'answer' and 'passages' keys. The runner handles this accordingly.
    """

    def __init__(self, name: str = "retrieval_task", dataset_name: Optional[str] = None):
        """Initialize retrieval task.
        
        Args:
            name: Task name.
            dataset_name: Optional dataset name to load specific prompts.
        """
        super().__init__(name)
        self.dataset_name = dataset_name
        self.system_prompt = self._get_system_prompt()
        self.response_schema = MSMARCO_RESPONSE_SCHEMA

    def _get_system_prompt(self) -> str:
        """Get system prompt from prompts registry based on dataset name.
        
        Returns:
            System prompt string.
        """
        if not self.dataset_name:
            return DEFAULT_RAG_PROMPT

        # Extract base dataset name (e.g., 'msmarco:validation' -> 'msmarco')
        base_dataset = self.dataset_name.split(":")[0].lower()
        
        return RAG_PROMPTS.get(base_dataset, DEFAULT_RAG_PROMPT)

    def build_prompt(self, example: Dict[str, Any], **kwargs) -> "GenerationInput":
        """Build GenerationInput with indexed passages as context.
        
        The context field should already be pre-formatted with [Passage N] prefixes
        by the MSMARCODataset loader.
        
        Includes response_schema for native structured output support.
        
        Args:
            example: Example dict with 'context' (indexed passages) and 'question'.
            **kwargs: Additional generation parameters.
            
        Returns:
            GenerationInput object with response_schema set.
        """
        from models.base import GenerationInput

        # See tasks/rag_task.py for why this pop is needed.
        kwargs.pop("model_mode", None)

        question = example.get("question", "")
        context = example.get("context", "")
        
        return GenerationInput(
            system_prompt=self.system_prompt,
            context=f"Passages:\n{context}",
            user_prompt=f"Question: {question}",
            response_schema=self.response_schema,
            **kwargs
        )

    def postprocess(self, output: str) -> Union[str, Dict[str, Any]]:
        """Parse structured output from model response.
        
        Uses native JSON parsing since models return structured output.
        Falls back to regex extraction if parsing fails.
        
        Args:
            output: Raw model output text (expected to be JSON).
            
        Returns:
            Dict with 'answer' (str or None) and 'passages' (List[int] or None).
        """
        return parse_msmarco_response(output)

    def normalize(self, text: str) -> str:
        """Normalize text for fair comparison.
        
        Uses default normalization from base Task class.
        
        Args:
            text: Text to normalize.
            
        Returns:
            Normalized text.
        """
        return super().normalize(text)
