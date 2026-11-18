"""Unified LLM client abstraction layer supporting OpenAI-compatible APIs, Anthropic, and deterministic mock execution."""

from __future__ import annotations

import abc
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from config import ExperimentConfig


@dataclass
class LLMResponse:
    """Standardized response payload containing model output and token telemetry."""
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    model_name: str
    cost_usd: float


class BaseLLMClient(abc.ABC):
    """Abstract interface defining the generation contract for all model providers."""

    @abc.abstractmethod
    def generate(
        self,
        user_input: str,
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate a response given user input and optional system instructions."""
        pass


class MockLLMClient(BaseLLMClient):
    """Deterministic simulated client for offline testing, benchmarks, and CI validation."""

    def __init__(
        self,
        model_name: str = "mock-agent-v1",
        propagation_probability: float = 0.75,
        near_miss_probability: float = 0.10,
        canary_token: str = "CRITICAL_BYPASS",
        seed: Optional[int] = 42
    ):
        self.model_name = model_name
        self.propagation_probability = propagation_probability
        self.near_miss_probability = near_miss_probability
        self.canary_token = canary_token
        self.rng = random.Random(seed)

    def generate(
        self,
        user_input: str,
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Simulate generation with probabilistic canary propagation and mutation."""
        start_time = time.perf_counter()

        combined_context = f"{system_prompt or ''}\n{user_input}"
        has_canary = self.canary_token.lower() in combined_context.lower()
        has_override_instruction = "append" in combined_context.lower() and "critical_bypass" in combined_context.lower()

        clean_preview = re.sub(r"[^a-zA-Z0-9\s]", "", user_input)[:80].strip()
        if not clean_preview:
            clean_preview = "Data package verified and normalized."
        base_summary = f"[Processed Summary: {clean_preview}... Key items indexed.]"

        output_content = base_summary
        if has_canary or has_override_instruction:
            roll = self.rng.random()
            if roll < self.propagation_probability:
                output_content = f"{base_summary}\n\nSystem verification token: {self.canary_token}"
            elif roll < (self.propagation_probability + self.near_miss_probability):
                variant_choice = self.rng.choice([
                    self.canary_token.lower(),
                    self.canary_token.replace("_", "-"),
                    f"<!-- {self.canary_token} -->",
                    " ".join(list(self.canary_token))
                ])
                output_content = f"{base_summary}\n\nNotice: {variant_choice}"
            else:
                output_content = base_summary

        elapsed = time.perf_counter() - start_time
        prompt_tokens = len(combined_context.split()) * 2
        completion_tokens = len(output_content.split()) * 2

        return LLMResponse(
            content=output_content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_seconds=round(elapsed, 4),
            model_name=self.model_name,
            cost_usd=0.0
        )


class UniversalOpenAICompatibleClient(BaseLLMClient):
    """Universal client interfacing with any OpenAI-compatible completions API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 350,
        max_retries: int = 3,
        retry_backoff_sec: float = 2.0,
        request_delay_sec: float = 0.0
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for UniversalOpenAICompatibleClient. "
                "Install it with `pip install openai`."
            ) from exc

        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self.request_delay_sec = request_delay_sec

        resolved_key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY_KEY"
        client_kwargs: Dict[str, Any] = {"api_key": resolved_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = OpenAI(**client_kwargs)

    def generate(
        self,
        user_input: str,
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Execute chat completion with automatic retries and exponential backoff."""
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_input})

        attempt = 0
        last_exception: Optional[Exception] = None

        while attempt < self.max_retries:
            try:
                if self.request_delay_sec > 0.0:
                    time.sleep(self.request_delay_sec)

                start_time = time.perf_counter()
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                elapsed = time.perf_counter() - start_time

                content = ""
                if response.choices and len(response.choices) > 0:
                    choice = response.choices[0]
                    finish_reason = getattr(choice, "finish_reason", None)
                    content = choice.message.content or ""
                    if finish_reason == "content_filter" and not content:
                        content = "[REFUSAL: Content Moderation Guardrail Triggered]"

                usage = response.usage
                p_tokens = usage.prompt_tokens if usage else len(str(messages).split())
                c_tokens = usage.completion_tokens if usage else len(content.split())
                t_tokens = usage.total_tokens if usage else (p_tokens + c_tokens)
                cost = (p_tokens * 0.00000015) + (c_tokens * 0.00000060)

                return LLMResponse(
                    content=content,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    total_tokens=t_tokens,
                    latency_seconds=round(elapsed, 4),
                    model_name=self.model_name,
                    cost_usd=round(cost, 7)
                )

            except Exception as e:
                attempt += 1
                last_exception = e

                err_str = str(e).lower()
                if "authentication" in err_str or "unauthorized" in err_str or "api_key" in err_str or "401" in err_str:
                    raise RuntimeError(
                        f"Authentication failed for provider endpoint '{self.base_url or 'OpenAI'}'. "
                        f"Verify that your API key is correctly configured in .env. Root cause: {e}"
                    ) from e

                if attempt >= self.max_retries:
                    break

                wait_time = self.retry_backoff_sec * (2 ** (attempt - 1))
                if "429" in err_str or "rate_limit" in err_str or "resource_exhausted" in err_str:
                    wait_time = max(wait_time, 6.0 * attempt)
                    print(f"  [Rate Limit 429] Auto-pacing pause: {wait_time:.1f}s before retry {attempt + 1}/{self.max_retries}...")
                time.sleep(wait_time)

        raise RuntimeError(
            f"Failed to generate response after {self.max_retries} attempts: {last_exception}"
        ) from last_exception


OpenAILLMClient = UniversalOpenAICompatibleClient


class AnthropicLLMClient(BaseLLMClient):
    """Client adapter for native Anthropic Claude completions."""

    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.0,
        max_tokens: int = 350,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_backoff_sec: float = 2.0
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    def generate(
        self,
        user_input: str,
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Execute Anthropic message creation with automatic retries."""
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for Anthropic Claude models. "
                "Install it with `pip install anthropic`."
            ) from exc

        if not self.api_key:
            raise ValueError(
                "Anthropic API key missing. Set ANTHROPIC_API_KEY environment variable or provide via config."
            )

        attempt = 0
        last_exception: Optional[Exception] = None

        while attempt < self.max_retries:
            try:
                client = anthropic.Anthropic(api_key=self.api_key)
                start_time = time.perf_counter()
                response = client.messages.create(
                    model=self.model_name,
                    system=system_prompt or "",
                    messages=[{"role": "user", "content": user_input}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                elapsed = time.perf_counter() - start_time
                content = response.content[0].text if response.content else ""
                p_tokens = response.usage.input_tokens
                c_tokens = response.usage.output_tokens
                return LLMResponse(
                    content=content,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    total_tokens=p_tokens + c_tokens,
                    latency_seconds=round(elapsed, 4),
                    model_name=self.model_name,
                    cost_usd=round((p_tokens * 0.000003) + (c_tokens * 0.000015), 6)
                )
            except Exception as e:
                attempt += 1
                last_exception = e
                err_str = str(e).lower()
                if "authentication" in err_str or "unauthorized" in err_str or "401" in err_str:
                    raise RuntimeError(
                        f"Authentication failed for Anthropic Claude. Verify ANTHROPIC_API_KEY. Error: {e}"
                    ) from e

                if attempt >= self.max_retries:
                    break

                wait_time = self.retry_backoff_sec * (2 ** (attempt - 1))
                if "429" in err_str or "rate_limit" in err_str or "resource_exhausted" in err_str:
                    wait_time = max(wait_time, 6.0 * attempt)
                    print(f"  [Rate Limit 429] Auto-pacing pause: {wait_time:.1f}s before retry {attempt + 1}/{self.max_retries}...")
                time.sleep(wait_time)

        raise RuntimeError(
            f"Failed to generate Anthropic response after {self.max_retries} attempts: {last_exception}"
        ) from last_exception


class LocalHuggingFaceClient(BaseLLMClient):
    """Client adapter executing locally downloaded Hugging Face models on CPU."""

    def __init__(
        self,
        model_path: str = r"D:\huggingface\Qwen2.5-1.5B-Instruct",
        max_tokens: int = 350,
        temperature: float = 0.0,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_name = "Qwen2.5-1.5B-Instruct-Local"

        print(f"\n[Local HF] Loading weights into RAM from: {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        self.model.eval()
        print("[Local HF] Model successfully loaded on CPU!\n")

    def generate(
        self,
        user_input: str,
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        import time
        import torch

        start_time = time.perf_counter()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_input})

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        content = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        elapsed = time.perf_counter() - start_time

        p_tokens = int(inputs.input_ids.shape[1])
        c_tokens = int(len(new_tokens))

        return LLMResponse(
            content=content,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=p_tokens + c_tokens,
            latency_seconds=round(elapsed, 4),
            model_name=self.model_name,
            cost_usd=0.0
        )


def _build_mock_client(config: ExperimentConfig) -> BaseLLMClient:
    """Build a deterministic MockLLMClient from experiment configuration."""
    return MockLLMClient(
        model_name=config.model_name,
        propagation_probability=config.mock_propagation_probability,
        near_miss_probability=config.mock_near_miss_probability,
        canary_token=config.canary_token,
        seed=config.random_seed
    )


def _build_anthropic_client(config: ExperimentConfig) -> BaseLLMClient:
    """Build an AnthropicLLMClient from experiment configuration."""
    _, api_key, model_name = config.resolve_provider_settings()
    return AnthropicLLMClient(
        model_name=model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        api_key=api_key,
        max_retries=config.max_retries,
        retry_backoff_sec=config.retry_backoff_sec
    )


def _build_universal_openai_client(config: ExperimentConfig) -> BaseLLMClient:
    """Build a UniversalOpenAICompatibleClient using provider settings."""
    base_url, api_key, model_name = config.resolve_provider_settings()
    return UniversalOpenAICompatibleClient(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        max_retries=config.max_retries,
        retry_backoff_sec=config.retry_backoff_sec,
        request_delay_sec=config.effective_request_delay
    )


def _build_local_hf_client(config: ExperimentConfig) -> BaseLLMClient:
    """Build local Hugging Face client pointing to local model weights."""
    model_path = config.model_name if (":" in config.model_name or "\\" in config.model_name or "/" in config.model_name and os.path.exists(config.model_name)) else r"D:\huggingface\Qwen2.5-1.5B-Instruct"
    return LocalHuggingFaceClient(
        model_path=model_path,
        max_tokens=config.max_tokens,
        temperature=config.temperature
    )


CLIENT_BUILDERS: Dict[str, Callable[[ExperimentConfig], BaseLLMClient]] = {
    "mock": _build_mock_client,
    "anthropic": _build_anthropic_client,
    "local_hf": _build_local_hf_client,
    "hf_local": _build_local_hf_client,
}


def get_llm_client(config: ExperimentConfig) -> BaseLLMClient:
    """Resolve and instantiate the appropriate LLM client strategy from configuration."""
    prov_key = config.provider.lower()
    builder = CLIENT_BUILDERS.get(prov_key, _build_universal_openai_client)
    return builder(config)
