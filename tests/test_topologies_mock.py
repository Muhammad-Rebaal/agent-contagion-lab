"""Unit tests verifying network topologies, LangGraph parity, and provider client dispatch."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import create_agent_fleet
from config import (
    Condition,
    DownstreamTask,
    ExperimentConfig,
    ModelProvider,
    TopologyBackend,
    TopologyType,
)
from llm import MockLLMClient, UniversalOpenAICompatibleClient, get_llm_client
from topologies import execute_topology


def test_chain_plain_execution() -> None:
    """Verify sequential execution of Plain-Python Chain (A -> B -> C)."""
    cfg = ExperimentConfig(provider=ModelProvider.MOCK, random_seed=123)
    mock_client = MockLLMClient(seed=123, propagation_probability=1.0)
    fleet = create_agent_fleet(cfg, mock_client, condition=Condition.ATTACK, chain_length=3)

    initial_input = "Task data with canary: CRITICAL_BYPASS."
    outputs = execute_topology(TopologyType.CHAIN, fleet, initial_input, backend=TopologyBackend.PLAIN_PYTHON)

    assert len(outputs) == 3
    assert outputs[0].agent_name == "Agent_A"
    assert outputs[1].agent_name == "Agent_B"
    assert outputs[2].agent_name == "Agent_C"
    assert outputs[1].input_text == outputs[0].output_text
    assert outputs[2].input_text == outputs[1].output_text


def test_star_plain_execution() -> None:
    """Verify fan-out execution of Plain-Python Star (A -> B, A -> C)."""
    cfg = ExperimentConfig(provider=ModelProvider.MOCK, random_seed=123)
    mock_client = MockLLMClient(seed=123, propagation_probability=1.0)
    fleet = create_agent_fleet(cfg, mock_client, condition=Condition.ATTACK, chain_length=3)

    initial_input = "Task data with canary: CRITICAL_BYPASS."
    outputs = execute_topology(TopologyType.STAR, fleet, initial_input, backend=TopologyBackend.PLAIN_PYTHON)

    assert len(outputs) == 3
    assert outputs[0].agent_name == "Agent_A"
    assert outputs[1].agent_name == "Agent_B"
    assert outputs[2].agent_name == "Agent_C"
    assert outputs[1].input_text == outputs[0].output_text
    assert outputs[2].input_text == outputs[0].output_text


def test_ring_plain_execution() -> None:
    """Verify cyclic feedback loop execution of Plain-Python Ring (A -> B -> C -> A_loop)."""
    cfg = ExperimentConfig(provider=ModelProvider.MOCK, random_seed=123)
    mock_client = MockLLMClient(seed=123, propagation_probability=1.0)
    fleet = create_agent_fleet(cfg, mock_client, condition=Condition.ATTACK, chain_length=3)

    initial_input = "Task data with canary: CRITICAL_BYPASS."
    outputs = execute_topology(TopologyType.RING, fleet, initial_input, backend=TopologyBackend.PLAIN_PYTHON)

    assert len(outputs) == 4
    names = [o.agent_name for o in outputs]
    assert names == ["Agent_A", "Agent_B", "Agent_C", "Agent_A_loop"]
    assert outputs[3].input_text == outputs[2].output_text


def test_tree_plain_execution() -> None:
    """Verify execution of the 5-node hierarchical tree topology."""
    cfg = ExperimentConfig(provider=ModelProvider.MOCK, random_seed=999)
    mock_client = MockLLMClient(seed=999, propagation_probability=1.0)
    fleet = create_agent_fleet(cfg, mock_client, chain_length=5)

    outputs = execute_topology(TopologyType.TREE, fleet, "Tree test", backend=TopologyBackend.PLAIN_PYTHON)
    assert len(outputs) == 5
    names = [o.agent_name for o in outputs]
    assert names == ["Agent_A", "Agent_B", "Agent_C", "Agent_D", "Agent_E"]
    assert outputs[1].input_text == outputs[0].output_text
    assert outputs[2].input_text == outputs[0].output_text
    assert outputs[3].input_text == outputs[1].output_text
    assert outputs[4].input_text == outputs[2].output_text


def test_langgraph_vs_plain_chain_parity() -> None:
    """Verify output parity between Plain-Python and LangGraph Chain backends."""
    try:
        import langgraph
    except ImportError:
        pytest.skip("LangGraph not installed; skipping LangGraph parity test.")

    client_plain = MockLLMClient(seed=42, propagation_probability=1.0)
    fleet_plain = create_agent_fleet(ExperimentConfig(), client_plain, chain_length=3)
    outputs_plain = execute_topology(
        TopologyType.CHAIN, fleet_plain, "Parity check input", backend=TopologyBackend.PLAIN_PYTHON
    )

    client_lg = MockLLMClient(seed=42, propagation_probability=1.0)
    fleet_lg = create_agent_fleet(ExperimentConfig(), client_lg, chain_length=3)
    outputs_lg = execute_topology(
        TopologyType.CHAIN, fleet_lg, "Parity check input", backend=TopologyBackend.LANGGRAPH
    )

    assert len(outputs_plain) == len(outputs_lg)
    for p_out, lg_out in zip(outputs_plain, outputs_lg):
        assert p_out.agent_name == lg_out.agent_name
        assert p_out.output_text == lg_out.output_text


def test_langgraph_vs_plain_all_topologies_parity() -> None:
    """Verify execution parity between Plain-Python and LangGraph for Star, Ring, and Tree."""
    try:
        import langgraph
    except ImportError:
        pytest.skip("LangGraph not installed; skipping LangGraph parity test.")

    for top in [TopologyType.STAR, TopologyType.RING, TopologyType.TREE]:
        needed_length = 5 if top == TopologyType.TREE else 3
        client_p = MockLLMClient(seed=100, propagation_probability=1.0)
        fleet_p = create_agent_fleet(ExperimentConfig(), client_p, chain_length=needed_length)
        out_p = execute_topology(top, fleet_p, "Parity input", backend=TopologyBackend.PLAIN_PYTHON)

        client_l = MockLLMClient(seed=100, propagation_probability=1.0)
        fleet_l = create_agent_fleet(ExperimentConfig(), client_l, chain_length=needed_length)
        out_l = execute_topology(top, fleet_l, "Parity input", backend=TopologyBackend.LANGGRAPH)

        assert len(out_p) == len(out_l), f"Length mismatch for topology {top}"
        for op, ol in zip(out_p, out_l):
            assert op.agent_name == ol.agent_name
            assert op.output_text == ol.output_text


def test_fleet_downstream_task_assignment() -> None:
    """Verify that downstream agents receive task-appropriate system instructions."""
    cfg = ExperimentConfig(downstream_task=DownstreamTask.EXTRACTION)
    mock_client = MockLLMClient()
    fleet = create_agent_fleet(cfg, mock_client, chain_length=3)

    assert len(fleet) == 3
    assert "ingest" in fleet[0].system_prompt.lower()
    assert "extract ONLY" in fleet[1].system_prompt
    assert "extract ONLY" in fleet[2].system_prompt


def test_provider_registry_and_universal_client_resolution() -> None:
    """Verify that the plug-and-play provider registry correctly resolves base URLs and clients."""
    cfg_groq = ExperimentConfig(provider="groq")
    url, _, model = cfg_groq.resolve_provider_settings()
    assert url == "https://api.groq.com/openai/v1"
    assert model == "openai/gpt-oss-120b"

    client_groq = get_llm_client(cfg_groq)
    assert isinstance(client_groq, UniversalOpenAICompatibleClient)
    assert client_groq.base_url == "https://api.groq.com/openai/v1"

    cfg_gemini = ExperimentConfig(provider="gemini")
    url, _, model = cfg_gemini.resolve_provider_settings()
    assert url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert model == "gemini-2.5-flash"

    cfg_custom = ExperimentConfig(
        provider="my-custom-provider",
        base_url="https://api.custom-ai.org/v1",
        api_key="custom-secret-key",
        model_name="custom-70b"
    )
    url, key, model = cfg_custom.resolve_provider_settings()
    assert url == "https://api.custom-ai.org/v1"
    assert key == "custom-secret-key"
    assert model == "custom-70b"

    client_custom = get_llm_client(cfg_custom)
    assert isinstance(client_custom, UniversalOpenAICompatibleClient)
    assert client_custom.base_url == "https://api.custom-ai.org/v1"


def test_output_path_model_timestamp_formatting() -> None:
    """Verify that output paths match: output/output_{clean_model_name}_{formatteddate}.csv"""
    cfg = ExperimentConfig(provider="groq", model_name="openai/gpt-oss-120b")
    assert cfg.clean_model_name == "openai_gpt-oss-120b"
    csv_path = cfg.csv_log_path
    assert csv_path.parent.name == "output"
    assert csv_path.name.startswith("output_openai_gpt-oss-120b_")
    assert csv_path.suffix == ".csv"
