"""OpenRouter model client.

OpenRouter exposes an OpenAI-compatible Chat Completions API that proxies to
many upstream providers, including DeepSeek. Model names use OpenRouter's
"provider/model" convention (e.g. "deepseek/deepseek-v4-flash").

Reasoning is disabled by default. DeepSeek V4 Flash/Pro on OpenRouter default
to an extended-thinking mode that spends completion tokens on a hidden
`reasoning` field before (or instead of, if max_tokens runs out first) the
visible `content` — confirmed live: a 10-token budget with reasoning enabled
returned `content: null` and `finish_reason: "length"`, all 10 tokens burned
on reasoning. That silently truncates low-max_tokens tasks (e.g. Classify)
and isn't apples-to-apples against non-reasoning competitor models on the
other benchmark pages (e.g. GPT-5.4 Mini). Pass reasoning_enabled=True to
this client only if you deliberately want to benchmark the reasoning variant.

Environment Variables:
    OPENROUTER_API_KEY: Required API key for OpenRouter.
"""
import logging
import time
from typing import Optional

from openai import OpenAI, RateLimitError, APIError, APIConnectionError

from models.base import ModelClient, GenerationInput, ModelOutput, PricingInfo

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(ModelClient):
    """Client for OpenRouter-routed models via the OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        pricing: Optional[PricingInfo] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        reasoning_enabled: bool = False,
    ):
        """Initialize OpenRouter client.

        Args:
            model_name: OpenRouter model id (e.g., "deepseek/deepseek-v4-flash").
            api_key: OpenRouter API key. If None, uses OPENROUTER_API_KEY env var.
            pricing: Optional pricing information for cost calculation.
            max_retries: Maximum number of retries for failed requests.
            retry_delay: Initial delay in seconds for exponential backoff.
            reasoning_enabled: Whether to allow the upstream model's extended
                thinking/reasoning mode. Default False — see module docstring.
        """
        super().__init__(model_name, api_key, pricing)
        self.provider = "openrouter"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.reasoning_enabled = reasoning_enabled

        import os
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenRouter API key required: pass api_key or set OPENROUTER_API_KEY."
            )
        self.client = OpenAI(api_key=resolved_key, base_url=OPENROUTER_BASE_URL)

    def generate(self, inp: GenerationInput) -> ModelOutput:
        """Generate text using OpenRouter's Chat Completions API.

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
            "extra_body": {"reasoning": {"enabled": self.reasoning_enabled}},
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
            params["response_format"] = {"type": "json_object"}

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(**params)
                latency_ms = (time.time() - start_time) * 1000

                if not response.choices:
                    raise RuntimeError(
                        f"OpenRouter returned no choices (response id={getattr(response, 'id', '?')}); "
                        "seen on at least one LongExtractBench example with a very large document — "
                        "root cause not yet identified (context window is 1M tokens, well above what "
                        "was sent, so it isn't a length issue). Not retried indefinitely by design: "
                        "the outer retry loop still gets 3 attempts, but if it fails all 3 it's treated "
                        "as a real per-example failure rather than transient."
                    )
                text = response.choices[0].message.content or ""
                input_tokens = response.usage.prompt_tokens if response.usage else 0
                output_tokens = response.usage.completion_tokens if response.usage else 0

                # Prefer OpenRouter's actual reported cost (it varies by
                # upstream provider routing) over our static PricingInfo when
                # available.
                actual_cost = getattr(response.usage, "cost", None) if response.usage else None
                cost_usd = actual_cost if actual_cost is not None else self._calculate_cost(input_tokens, output_tokens)

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
                logger.error(f"Unexpected error in OpenRouterClient.generate: {e}", exc_info=True)
                raise
        if last_exception is not None:
            raise last_exception
        else:
            raise RuntimeError("OpenRouterClient.generate failed with an unexpected error and no exception was captured.")

    def count_tokens(self, text: str) -> int:
        """Approximate token count (chars / 4) — OpenRouter has no single tokenizer
        since it proxies many upstream models. Actual cost accounting always uses
        the `usage` block returned by the API in generate()."""
        return max(1, len(text) // 4)
