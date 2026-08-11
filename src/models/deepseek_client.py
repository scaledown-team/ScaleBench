"""DeepSeek model client.

DeepSeek exposes an OpenAI-compatible Chat Completions API, so this client
reuses the `openai` SDK pointed at DeepSeek's base URL rather than a bespoke
HTTP layer. Token usage for cost calculation comes straight from the API
response (DeepSeek has no public tokenizer package for local counting).

Environment Variables:
    DEEPSEEK_API_KEY: Required API key for DeepSeek.

Model names (verify against your account before relying on these — DeepSeek
was mid-rollout on V4 Pro as of Aug 2026, see PR description):
    deepseek-v4-flash
    deepseek-v4-pro
"""
import logging
import time
from typing import Optional

from openai import OpenAI, RateLimitError, APIError, APIConnectionError

from models.base import ModelClient, GenerationInput, ModelOutput, PricingInfo

logger = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekClient(ModelClient):
    """Client for DeepSeek models via the OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        pricing: Optional[PricingInfo] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """Initialize DeepSeek client.

        Args:
            model_name: DeepSeek model name (e.g., "deepseek-v4-flash", "deepseek-v4-pro").
            api_key: DeepSeek API key. If None, uses DEEPSEEK_API_KEY env var.
            pricing: Optional pricing information for cost calculation.
            max_retries: Maximum number of retries for failed requests.
            retry_delay: Initial delay in seconds for exponential backoff.
        """
        super().__init__(model_name, api_key, pricing)
        self.provider = "deepseek"
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # DEEPSEEK_API_KEY is read by the OpenAI SDK only if api_key is passed
        # explicitly here; there is no implicit env var fallback like OPENAI_API_KEY.
        import os
        resolved_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not resolved_key:
            raise ValueError(
                "DeepSeek API key required: pass api_key or set DEEPSEEK_API_KEY."
            )
        self.client = OpenAI(api_key=resolved_key, base_url=DEEPSEEK_BASE_URL)

    def generate(self, inp: GenerationInput) -> ModelOutput:
        """Generate text using DeepSeek's Chat Completions API.

        Args:
            inp: GenerationInput with prompt and configuration.

        Returns:
            ModelOutput with generated text and metadata.
        """
        messages = []
        if inp.system_prompt:
            messages.append({"role": "system", "content": inp.system_prompt})

        user_content = inp.user_prompt
        if inp.context:
            user_content = f"{inp.context}\n\n{inp.user_prompt}"
        messages.append({"role": "user", "content": user_content})

        params = {
            "model": self.model_name,
            "messages": messages,
        }
        if inp.temperature is not None:
            params["temperature"] = inp.temperature
        if inp.max_output_tokens is not None:
            params["max_tokens"] = inp.max_output_tokens
        if inp.top_p is not None:
            params["top_p"] = inp.top_p
        if inp.stop is not None:
            params["stop"] = inp.stop
        if inp.response_schema is not None:
            # DeepSeek supports {"type": "json_object"} but not strict json_schema
            # mode as of Aug 2026 — fall back to json_object and rely on the
            # system prompt to describe the schema.
            params["response_format"] = {"type": "json_object"}

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(**params)
                latency_ms = (time.time() - start_time) * 1000

                text = response.choices[0].message.content or ""
                input_tokens = response.usage.prompt_tokens if response.usage else 0
                output_tokens = response.usage.completion_tokens if response.usage else 0
                cost_usd = self._calculate_cost(input_tokens, output_tokens)

                return ModelOutput(
                    text=text,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    input_question=inp.user_prompt,
                    input_context=inp.context,
                    input_system_prompt=inp.system_prompt,
                )

            except RateLimitError as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(delay)
                else:
                    logger.error(f"Rate limit exceeded after {self.max_retries} retries")

            except (APIError, APIConnectionError) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"API error, retrying in {delay}s (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"API error after {self.max_retries} retries: {e}")
            except Exception as e:
                logger.error(f"Unexpected error in DeepSeekClient.generate: {e}", exc_info=True)
                raise
        if last_exception is not None:
            raise last_exception
        else:
            raise RuntimeError("DeepSeekClient.generate failed with an unexpected error and no exception was captured.")

    def count_tokens(self, text: str) -> int:
        """Approximate token count for text not yet sent to the API.

        DeepSeek has no public local tokenizer, so this is a rough heuristic
        (chars / 4) used only for pre-flight estimates. Actual cost accounting
        always uses the `usage` block returned by the API in generate().

        Args:
            text: Input text to count tokens for.

        Returns:
            Approximate token count.
        """
        return max(1, len(text) // 4)
