"""ModelClient interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List


@dataclass
class PricingInfo:
    """Pricing information for a model."""
    
    input_per_1m_tokens: Optional[float] = None
    output_per_1m_tokens: Optional[float] = None
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> Optional[float]:
        """Calculate cost in USD based on token usage.
        
        Args:
            input_tokens: Number of input tokens used.
            output_tokens: Number of output tokens generated.
            
        Returns:
            Cost in USD, or None if pricing not configured.
        """
        if self.input_per_1m_tokens is None or self.output_per_1m_tokens is None:
            return None
        
        input_cost = (input_tokens / 1_000_000.0) * self.input_per_1m_tokens
        output_cost = (output_tokens / 1_000_000.0) * self.output_per_1m_tokens
        return input_cost + output_cost


# Default pricing for common models (per 1M tokens)
# Source: Provider pricing pages as of December 2025
DEFAULT_MODEL_PRICING: Dict[str, PricingInfo] = {
    # OpenAI models
    "gpt-4o": PricingInfo(input_per_1m_tokens=2.50, output_per_1m_tokens=10.00),
    "gpt-4o-mini": PricingInfo(input_per_1m_tokens=0.15, output_per_1m_tokens=0.60),
    "gpt-4.1": PricingInfo(input_per_1m_tokens=2.00, output_per_1m_tokens=8.00),
    "gpt-4.1-mini": PricingInfo(input_per_1m_tokens=0.40, output_per_1m_tokens=1.60),
    "gpt-5-mini": PricingInfo(input_per_1m_tokens=0.25, output_per_1m_tokens=2.00),
    "gpt-5.2": PricingInfo(input_per_1m_tokens=1.75, output_per_1m_tokens=14.00),
    "gpt-5.2-pro": PricingInfo(input_per_1m_tokens=21.00, output_per_1m_tokens=168.00),
    
    # Gemini models
    "gemini-2.5-flash": PricingInfo(input_per_1m_tokens=0.30, output_per_1m_tokens=2.50),
    "gemini-2.5-flash-lite": PricingInfo(input_per_1m_tokens=0.10, output_per_1m_tokens=0.40),
    "gemini-2.5-pro": PricingInfo(input_per_1m_tokens=1.25, output_per_1m_tokens=10.00),

    # DeepSeek models (cache-miss input rate; see deepseek.ai/pricing, Aug 2026)
    "deepseek-v4-flash": PricingInfo(input_per_1m_tokens=0.14, output_per_1m_tokens=0.28),
    "deepseek-v4-pro": PricingInfo(input_per_1m_tokens=0.435, output_per_1m_tokens=0.87),
}


@dataclass
class GenerationInput:
    """Input for model generation with configurable parameters."""
    
    user_prompt: str
    context: Optional[str] = None
    system_prompt: Optional[str] = None
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None
    response_schema: Optional[Dict[str, Any]] = None  # JSON Schema for structured output


class ModelOutput:
    """Container for model generation output."""

    def __init__(
        self,
        text: str,
        latency_ms: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: Optional[float] = None,
        provider_metadata: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        input_question: Optional[str] = None,
        input_context: Optional[str] = None,
        input_system_prompt: Optional[str] = None,
    ):
        self.text = text
        self.latency_ms = latency_ms
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost_usd = cost_usd
        self.provider_metadata = provider_metadata or {}
        # Keep metadata for backward compatibility
        self.metadata = metadata or {}
        self.input_question = input_question
        self.input_context = input_context
        self.input_system_prompt = input_system_prompt


class ModelClient(ABC):
    """Abstract base class for model clients."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        pricing: Optional[PricingInfo] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.pricing = pricing
        self.provider = "unknown"  # Override in subclasses

    @abstractmethod
    def generate(self, inp: GenerationInput) -> ModelOutput:
        """Generate text from input.
        
        Args:
            inp: GenerationInput with prompt and configuration.
            
        Returns:
            ModelOutput with generated text and metadata.
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Input text to count tokens for.
            
        Returns:
            Token count.
        """
        pass

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> Optional[float]:
        """Calculate cost if pricing is configured.
        
        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            
        Returns:
            Cost in USD or None.
        """
        if self.pricing is None:
            return None
        return self.pricing.calculate_cost(input_tokens, output_tokens)

    @property
    def name(self) -> str:
        """Model name."""
        return self.model_name
