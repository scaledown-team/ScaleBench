"""RAG task (e.g. MS MARCO / HotpotQA RAG)."""
from typing import Dict, Any, Optional
from tasks.base import Task
from prompts.rag_prompts import RAG_PROMPTS, DEFAULT_RAG_PROMPT


class RAGTask(Task):
    """RAG-style QA task for multi-hop reasoning."""

    def __init__(self, name: str = "rag_qa", dataset_name: Optional[str] = None):
        """Initialize RAG task.
        
        Args:
            name: Task name.
            dataset_name: Optional dataset name to load specific prompts.
        """
        super().__init__(name)
        self.dataset_name = dataset_name
        self.system_prompt = self._get_system_prompt()

    def _get_system_prompt(self) -> str:
        """Get system prompt from prompts registry based on dataset name.
        
        Returns:
            System prompt string.
        """
        if not self.dataset_name:
            return DEFAULT_RAG_PROMPT

        # Extract base dataset name (e.g., 'hotpotqa:distractor' -> 'hotpotqa')
        base_dataset = self.dataset_name.split(":")[0].lower()
        
        return RAG_PROMPTS.get(base_dataset, DEFAULT_RAG_PROMPT)

    def _is_ethic_dataset(self) -> bool:
        """Check if this task is configured for ETHIC dataset.
        
        Returns:
            True if dataset_name starts with 'ethic'.
        """
        if not self.dataset_name:
            return False
        return self.dataset_name.split(":")[0].lower() == "ethic"

    def _parse_ethic_user_msg(self, user_msg: str) -> tuple:
        """Parse ETHIC User_msg into context and instruction parts.
        
        ETHIC User_msg format:
            ### Context:
            <long document>
            ### Instruction:
            <task instruction>
            Answer:
        
        Args:
            user_msg: The full User_msg field from ETHIC dataset.
            
        Returns:
            Tuple of (context, instruction) strings.
        """
        context = ""
        instruction = ""
        
        # Split by ### Context: marker
        if "### Context:" in user_msg:
            parts = user_msg.split("### Context:", 1)
            rest = parts[1] if len(parts) > 1 else ""
            
            # Split by ### Instruction: marker
            if "### Instruction:" in rest:
                ctx_part, inst_part = rest.split("### Instruction:", 1)
                context = ctx_part.strip()
                instruction = inst_part.strip()
            else:
                # No instruction marker, treat entire rest as context
                context = rest.strip()
        else:
            # No context marker, use full message as instruction
            instruction = user_msg.strip()
        
        return context, instruction

    def build_prompt(self, example: Dict[str, Any], **kwargs) -> "GenerationInput":
        """Build GenerationInput with specific ordering: Instructions -> Context -> Question.
        
        For ETHIC dataset, extracts context from User_msg for compression while
        keeping the instruction as user_prompt.
        
        Args:
            example: Example with 'context' and 'question' fields.
            **kwargs: Additional generation parameters.
            
        Returns:
            GenerationInput object.
        """
        from models.base import GenerationInput

        # model_mode is passed by evaluation/runner.py for every task (only
        # SummarizationTask's ScaleDown-endpoint branch actually needs it);
        # pop it here so it doesn't reach GenerationInput(), which has no
        # such field and raises a TypeError on an unexpected kwarg.
        kwargs.pop("model_mode", None)

        question = example.get("question", "")
        context = example.get("context", "")
        
        # ETHIC: Parse User_msg to extract context and instruction separately
        # This allows ScaleDown to compress the context portion
        if self._is_ethic_dataset() and not question:
            ethic_context, ethic_instruction = self._parse_ethic_user_msg(context)
            return GenerationInput(
                system_prompt=self.system_prompt,
                context=f"### Context:\n{ethic_context}" if ethic_context else "",
                user_prompt=f"### Instruction:\n{ethic_instruction}",
                **kwargs
            )
        
        # Standard datasets: build structured prompt
        return GenerationInput(
            system_prompt=self.system_prompt,
            context=f"Context:\n{context}",
            user_prompt=f"Question: {question}\n\nAnswer:",
            **kwargs
        )

    def postprocess(self, output: str) -> str:
        """Extract answer from model output.
        
        For ETHIC dataset, returns full output for set-based metrics.
        For other datasets, takes first non-empty line as the answer.
        
        Args:
            output: Raw model output text.
            
        Returns:
            Extracted answer string.
        """
        # ETHIC: return full multi-line output for set metrics to parse
        if self._is_ethic_dataset():
            return output.strip()
        
        # Standard datasets: take first non-empty line
        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line:
                return line
        
        # Fallback to full output if no lines found
        return output.strip()
