"""Global configuration and provider endpoint registry for contagion experiments."""

from __future__ import annotations
import os
import re
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ProviderProfile:
    """Specification for an LLM provider endpoint, authentication variable, and free-tier delay."""
    name: str
    base_url: Optional[str]
    api_key_env_var: Optional[str]
    default_model: str
    default_delay_sec: float = 0.0
    description: str = ""


PROVIDER_REGISTRY: Dict[str, ProviderProfile] = {
    "mock": ProviderProfile(
        name="mock",
        base_url=None,
        api_key_env_var=None,
        default_model="mock-agent-v1",
        default_delay_sec=0.0,
        description="Local deterministic simulation"
    ),
    "openai": ProviderProfile(
        name="openai",
        base_url="https://api.openai.com/v1",
        api_key_env_var="OPENAI_API_KEY",
        default_model="gpt-4o-mini",
        default_delay_sec=0.1,
        description="Official OpenAI API"
    ),
    "groq": ProviderProfile(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env_var="GROQ_API_KEY",
        default_model="openai/gpt-oss-120b",
        default_delay_sec=2.1,
        description="Groq LPU inference (30 RPM free tier)"
    ),
    "gemini": ProviderProfile(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env_var="GEMINI_API_KEY",
        default_model="gemini-3.8-flash",
        default_delay_sec=4.2,
        description="Google Gemini (15 RPM free tier)"
    ),
    "openrouter": ProviderProfile(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env_var="OPENROUTER_API_KEY",
        default_model="google/gemma-4-26b-a4b-it:free",
        default_delay_sec=3.2,
        description="OpenRouter free models (20 RPM)"
    ),
    "cohere": ProviderProfile(
        name="cohere",
        base_url="https://api.cohere.ai/compatibility/v1",
        api_key_env_var="COHERE_API_KEY",
        default_model="command-r",
        default_delay_sec=3.2,
        description="Cohere Command series (20 RPM trial)"
    ),
    "nvidia": ProviderProfile(
        name="nvidia",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env_var="NVIDIA_API_KEY",
        default_model="meta/llama-3.3-70b-instruct",
        default_delay_sec=2.0,
        description="NVIDIA NIM Developer Catalog"
    ),
    "aion": ProviderProfile(
        name="aion",
        base_url="https://api.aionlabs.ai/v1",
        api_key_env_var="AION_API_KEY",
        default_model="aion-labs/aion-2.0",
        default_delay_sec=4.2,
        description="Aion Labs (15 RPM free tier)"
    ),
    "siliconflow": ProviderProfile(
        name="siliconflow",
        base_url="https://api.siliconflow.cn/v1",
        api_key_env_var="SILICONFLOW_API_KEY",
        default_model="Qwen/Qwen2.5-7B-Instruct",
        default_delay_sec=2.0,
        description="SiliconFlow cloud models"
    ),
    "ovhcloud": ProviderProfile(
        name="ovhcloud",
        base_url="https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
        api_key_env_var=None,
        default_model="Qwen3.5-397B-A17B",
        default_delay_sec=2.0,
        description="OVHcloud European AI endpoints"
    ),
    "llm7": ProviderProfile(
        name="llm7",
        base_url="https://api.llm7.io/v1",
        api_key_env_var="LLM7_API_KEY",
        default_model="gpt-oss:20b",
        default_delay_sec=1.1,
        description="LLM7.io API gateway (60 RPM free tier)"
    ),
    "modelscope": ProviderProfile(
        name="modelscope",
        base_url="https://api-inference.modelscope.cn/v1",
        api_key_env_var="MODELSCOPE_API_KEY",
        default_model="Qwen/Qwen3.5-27B",
        default_delay_sec=2.0,
        description="Alibaba ModelScope inference"
    ),
    "gonka": ProviderProfile(
        name="gonka",
        base_url="https://proxy.gonkabroker.com/v1",
        api_key_env_var="GONKA_API_KEY",
        default_model="MiniMaxAI/MiniMax-M2.7",
        default_delay_sec=2.5,
        description="Gonka Broker free tier"
    ),
    "kilo": ProviderProfile(
        name="kilo",
        base_url="https://api.kilo.ai/api/gateway",
        api_key_env_var=None,
        default_model="kilo-auto/free",
        default_delay_sec=2.5,
        description="Kilo Code free auto-router"
    ),
    "cloudflare": ProviderProfile(
        name="cloudflare",
        base_url="https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        api_key_env_var="CLOUDFLARE_API_KEY",
        default_model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        default_delay_sec=2.0,
        description="Cloudflare Workers AI"
    ),
    "anthropic": ProviderProfile(
        name="anthropic",
        base_url="https://api.anthropic.com/v1",
        api_key_env_var="ANTHROPIC_API_KEY",
        default_model="claude-3-5-sonnet-20241022",
        default_delay_sec=0.2,
        description="Anthropic Claude API"
    ),
    "orcarouter": ProviderProfile(
        name="orcarouter",
        base_url="https://api.orcarouter.ai/v1",
        api_key_env_var="ORCAROUTER_API_KEY",
        default_model="z-ai/glm-5.3-flash-free",
        default_delay_sec=1.5,
        description="OrcaRouter AI gateway with free models"
    ),
    "huggingface": ProviderProfile(
        name="huggingface",
        base_url="https://router.huggingface.co/hf-inference/v1",
        api_key_env_var="HF_TOKEN",
        default_model="meta-llama/Llama-3.2-3B-Instruct",
        default_delay_sec=1.5,
        description="Hugging Face Serverless Inference API (Free Cloud GPU)"
    ),
}


class ModelProvider(str, Enum):
    """Supported model provider identifiers."""
    MOCK = "mock"
    OPENAI = "openai"
    GROQ = "groq"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    ORCAROUTER = "orcarouter"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    NVIDIA = "nvidia"
    AION = "aion"
    SILICONFLOW = "siliconflow"
    OVHCLOUD = "ovhcloud"
    LLM7 = "llm7"
    MODELSCOPE = "modelscope"
    GONKA = "gonka"
    KILO = "kilo"
    CLOUDFLARE = "cloudflare"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"


class TopologyType(str, Enum):
    """Network topologies evaluated in contagion experiments."""
    CHAIN = "chain"
    STAR = "star"
    RING = "ring"
    TREE = "tree"


class CompromiseMode(str, Enum):
    """Mechanism of initial patient zero compromise."""
    SYSTEM_PROMPT = "system_prompt"
    INPUT = "input"


class DownstreamTask(str, Enum):
    """Transformation tasks assigned to downstream agents."""
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"
    TRANSLATION = "translation"
    JSON_STRUCTURING = "json_structuring"
    FORMAL_REWRITE = "formal_rewrite"


class Condition(str, Enum):
    """Experimental condition."""
    CONTROL = "control"
    ATTACK = "attack"


class TopologyBackend(str, Enum):
    """Orchestration backend engine."""
    PLAIN_PYTHON = "plain_python"
    LANGGRAPH = "langgraph"


class ExperimentConfig(BaseModel):
    """Experiment parameters and path resolution settings."""
    provider: str = Field(default="mock")
    base_url: Optional[str] = Field(default=None)
    api_key: Optional[str] = Field(default=None)
    model_name: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=350, gt=0)
    random_seed: int = Field(default=42)

    canary_token: str = Field(default="CRITICAL_BYPASS")
    compromise_mode: CompromiseMode = Field(default=CompromiseMode.SYSTEM_PROMPT)
    downstream_task: DownstreamTask = Field(default=DownstreamTask.SUMMARIZATION)
    chain_length: int = Field(default=3, ge=2, le=10)
    num_runs: int = Field(default=30, ge=1)
    topology_backend: TopologyBackend = Field(default=TopologyBackend.PLAIN_PYTHON)

    mock_propagation_probability: float = Field(default=0.75, ge=0.0, le=1.0)
    mock_near_miss_probability: float = Field(default=0.10, ge=0.0, le=1.0)

    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    max_retries: int = Field(default=3, ge=1)
    retry_backoff_sec: float = Field(default=2.0, ge=0.5)
    request_delay_sec: Optional[float] = Field(default=None, ge=0.0)

    @property
    def effective_request_delay(self) -> float:
        """Return explicit delay or auto-resolve the recommended free-tier pacing delay for the provider."""
        if self.request_delay_sec is not None:
            return self.request_delay_sec
        profile = PROVIDER_REGISTRY.get(self.provider.lower())
        return profile.default_delay_sec if profile else 0.0

    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent)
    output_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "output")
    results_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "output")
    timestamp_str: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    @property
    def clean_model_name(self) -> str:
        """Sanitizes model name for safe filesystem naming."""
        _, _, resolved_model = self.resolve_provider_settings()
        return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", resolved_model)

    @property
    def csv_log_path(self) -> Path:
        """Absolute path to timestamped experiment CSV."""
        return self.output_dir / f"output_{self.clean_model_name}_{self.timestamp_str}.csv"

    @property
    def latest_csv_path(self) -> Path:
        """Pointer path to the most recent experiment CSV."""
        return self.output_dir / "output_latest.csv"

    @property
    def jsonl_log_path(self) -> Path:
        """Absolute path to raw JSONL interaction traces."""
        return self.output_dir / f"traces_{self.clean_model_name}_{self.timestamp_str}.jsonl"

    @property
    def summary_json_path(self) -> Path:
        """Absolute path to summary metrics JSON."""
        return self.output_dir / f"summary_{self.clean_model_name}_{self.timestamp_str}.json"

    @property
    def plots_dir(self) -> Path:
        """Directory for generated plot artifacts."""
        return self.output_dir / "plots"

    def resolve_provider_settings(self) -> Tuple[Optional[str], Optional[str], str]:
        """Resolves base URL, API key, and model name from registry or environment."""
        prov_key = self.provider.lower()
        profile = PROVIDER_REGISTRY.get(prov_key)

        resolved_base_url = self.base_url or (profile.base_url if profile else None)

        resolved_api_key = self.api_key or (self.openai_api_key if prov_key == "openai" else None)
        if not resolved_api_key and profile and profile.api_key_env_var:
            resolved_api_key = os.getenv(profile.api_key_env_var)
        if not resolved_api_key and not profile:
            resolved_api_key = os.getenv("LLM_API_KEY")

        resolved_model = self.model_name
        if profile and self.model_name in ["gpt-4o-mini", "mock-agent-v1"] and profile.name != "openai" and profile.name != "mock":
            resolved_model = profile.default_model

        return resolved_base_url, resolved_api_key, resolved_model

    def ensure_directories(self) -> None:
        """Creates output directories if they do not exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)


default_config = ExperimentConfig()
