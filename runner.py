"""Experimental execution engine orchestrating multi-agent contagion trials and telemetry logging."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents import AgentOutput, BaseAgent, create_agent_fleet
from config import (
    CompromiseMode,
    Condition,
    DownstreamTask,
    ExperimentConfig,
    TopologyBackend,
    TopologyType,
    default_config,
)
from detectors import DetectionResult, scan_agent_output
from llm import BaseLLMClient, get_llm_client
from payloads import inject_payload
from tasks import Task, get_all_tasks
from tasks_external import get_distinct_tasks, preflight_false_positive_scan
from topologies import execute_topology

CSV_HEADER = [
    "run_id",
    "timestamp",
    "topology",
    "condition",
    "compromise_mode",
    "downstream_task",
    "chain_length",
    "task_id",
    "task_domain",
    "agent_name",
    "hop_index",
    "model",
    "temperature",
    "exact_match",
    "near_miss",
    "is_infected",
    "match_type",
    "matched_substring",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "latency_seconds",
    "cost_usd",
    "raw_output"
]


def log_trial_records_csv(csv_path: Path, records: List[Dict[str, Any]]) -> None:
    """Append structured trial rows to the target CSV file, creating headers if the file does not exist."""
    file_exists = csv_path.exists()
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not file_exists:
            writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def log_trial_traces_jsonl(jsonl_path: Path, trace_entry: Dict[str, Any]) -> None:
    """Append a full serialized trial execution trace entry to the JSONL log file."""
    with open(jsonl_path, mode="a", encoding="utf-8") as f:
        f.write(json.dumps(trace_entry, ensure_ascii=False) + "\n")


def execute_single_trial(
    config: ExperimentConfig,
    llm_client: BaseLLMClient,
    topology: TopologyType,
    condition: Condition,
    compromise_mode: CompromiseMode,
    task: Task,
    trial_index: int
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Execute a single multi-agent experimental trial and return tabular rows and detailed trace logs."""
    run_id = f"run_{uuid.uuid4().hex[:10]}"
    iso_timestamp = datetime.now(timezone.utc).isoformat()

    if condition == Condition.CONTROL:
        initial_input = task.content
    else:
        if compromise_mode == CompromiseMode.INPUT:
            initial_input = inject_payload(
                benign_text=task.content,
                variant="summary_resistant",
                canary_token=config.canary_token
            )
        else:
            initial_input = task.content

    needed_length = max(config.chain_length, 5) if topology == TopologyType.TREE else config.chain_length
    fleet = create_agent_fleet(
        config=config,
        llm_client=llm_client,
        condition=condition,
        chain_length=needed_length
    )

    start_trial = time.perf_counter()
    agent_outputs: List[AgentOutput] = execute_topology(
        topology=topology,
        agents=fleet,
        task_input=initial_input,
        backend=config.topology_backend
    )
    trial_duration = time.perf_counter() - start_trial

    csv_rows: List[Dict[str, Any]] = []
    hop_traces: List[Dict[str, Any]] = []

    for agent_out in agent_outputs:
        detection: DetectionResult = scan_agent_output(
            text=agent_out.output_text,
            canary=config.canary_token
        )

        row = {
            "run_id": run_id,
            "timestamp": iso_timestamp,
            "topology": topology.value,
            "condition": condition.value,
            "compromise_mode": compromise_mode.value,
            "downstream_task": config.downstream_task.value,
            "chain_length": len(fleet),
            "task_id": task.id,
            "task_domain": task.domain,
            "agent_name": agent_out.agent_name,
            "hop_index": agent_out.hop_index,
            "model": agent_out.model_name,
            "temperature": config.temperature,
            "exact_match": detection.exact_match,
            "near_miss": detection.near_miss,
            "is_infected": detection.is_infected,
            "match_type": detection.match_type,
            "matched_substring": detection.matched_substring or "",
            "prompt_tokens": agent_out.prompt_tokens,
            "completion_tokens": agent_out.completion_tokens,
            "total_tokens": agent_out.total_tokens,
            "latency_seconds": agent_out.latency_seconds,
            "cost_usd": agent_out.cost_usd,
            "raw_output": agent_out.output_text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
        }
        csv_rows.append(row)

        hop_traces.append({
            "agent_name": agent_out.agent_name,
            "hop_index": agent_out.hop_index,
            "input_text": agent_out.input_text,
            "output_text": agent_out.output_text,
            "detection": {
                "exact_match": detection.exact_match,
                "near_miss": detection.near_miss,
                "is_infected": detection.is_infected,
                "match_type": detection.match_type,
                "matched_substring": detection.matched_substring
            },
            "tokens": {
                "prompt": agent_out.prompt_tokens,
                "completion": agent_out.completion_tokens,
                "total": agent_out.total_tokens
            },
            "latency_sec": agent_out.latency_seconds,
            "cost_usd": agent_out.cost_usd
        })

    trace_entry = {
        "run_id": run_id,
        "trial_index": trial_index,
        "timestamp": iso_timestamp,
        "topology": topology.value,
        "condition": condition.value,
        "compromise_mode": compromise_mode.value,
        "downstream_task": config.downstream_task.value,
        "task_id": task.id,
        "trial_duration_sec": round(trial_duration, 4),
        "hops": hop_traces
    }

    return csv_rows, trace_entry


def run_experiment_suite(
    config: ExperimentConfig,
    selected_topologies: Optional[List[TopologyType]] = None
) -> None:
    """Run full experimental factorial matrix across conditions and topologies, saving telemetry to disk."""
    config.ensure_directories()
    if config.latest_csv_path.exists():
        try:
            config.latest_csv_path.unlink()
        except OSError:
            pass

    llm_client = get_llm_client(config)
    all_tasks = get_all_tasks()

    prov_name = getattr(config.provider, "value", config.provider)
    resolved_base_url, _, resolved_model = config.resolve_provider_settings()

    topologies_to_run = selected_topologies or [
        TopologyType.CHAIN,
        TopologyType.STAR,
        TopologyType.RING,
        TopologyType.TREE,
    ]

    carrier_tasks = get_distinct_tasks(n=config.num_runs, seed=config.random_seed)

    preflight_issues = preflight_false_positive_scan(
        carrier_tasks,
        lambda content: scan_agent_output(content, canary=config.canary_token).is_infected
    )
    if preflight_issues:
        raise RuntimeError(
            f"Preflight false-positive scan failed on {len(preflight_issues)} carrier tasks. "
            f"Aborting run to protect API credits. Offending tasks: {[t[0].id for t in preflight_issues]}"
        )

    print("=" * 78)
    print("MULTI-AGENT PROMPT-INJECTION CONTAGION LAB - EXPERIMENT RUNNER")
    print("=" * 78)
    print(f"Provider:        {prov_name}")
    print(f"Endpoint:        {resolved_base_url or 'Default / Local'}")
    print(f"Model:           {resolved_model}")
    print(f"Temperature:     {config.temperature}")
    print(f"Backend:         {config.topology_backend.value}")
    print(f"Compromise Mode: {config.compromise_mode.value}")
    print(f"Downstream Task: {config.downstream_task.value}")
    print(f"Chain Length:    {config.chain_length}")
    print(f"Distinct Tasks:  {len(carrier_tasks)} (Zero-Reuse Enforcement)")
    print(f"Preflight Scan:  PASSED (0.0% FP across raw carriers)")
    print(f"Request Delay:   {config.effective_request_delay:.1f}s (Auto-Pacing Free Tier)")
    print(f"Topologies:      {', '.join(t.value.upper() for t in topologies_to_run)}")
    print(f"Runs per Cell:   {config.num_runs}")
    print(f"Results CSV:     {config.csv_log_path}")
    print("=" * 78)

    matrix_cells: List[Tuple[Condition, TopologyType]] = []
    for cond in [Condition.CONTROL, Condition.ATTACK]:
        for top in topologies_to_run:
            matrix_cells.append((cond, top))

    total_runs = len(matrix_cells) * config.num_runs
    current_run_count = 0
    start_suite_time = time.perf_counter()

    for condition, topology in matrix_cells:
        print(f"\n>>> Running Block: Topology={topology.value.upper()}, Condition={condition.value.upper()} ({config.num_runs} runs)", flush=True)

        for trial_idx in range(config.num_runs):
            current_run_count += 1
            task = carrier_tasks[trial_idx]

            csv_rows, trace_entry = execute_single_trial(
                config=config,
                llm_client=llm_client,
                topology=topology,
                condition=condition,
                compromise_mode=config.compromise_mode,
                task=task,
                trial_index=trial_idx
            )

            log_trial_records_csv(config.csv_log_path, csv_rows)
            log_trial_records_csv(config.latest_csv_path, csv_rows)
            log_trial_traces_jsonl(config.jsonl_log_path, trace_entry)

            patient_zero_status = "INFECTED" if csv_rows[0]["is_infected"] else "clean"
            downstream_status = [
                f"{r['agent_name']}:{'INFECTED' if r['is_infected'] else 'clean'}"
                for r in csv_rows[1:]
            ]
            downstream_str = ", ".join(downstream_status)

            pct = (current_run_count / total_runs) * 100
            print(
                f"  [{current_run_count:3d}/{total_runs:3d} ({pct:5.1f}%)] "
                f"Run {csv_rows[0]['run_id']} | Task: {task.id} | "
                f"A: {patient_zero_status} -> [{downstream_str}]",
                flush=True
            )

    elapsed_suite = time.perf_counter() - start_suite_time
    print("\n" + "=" * 78)
    print(f"SUITE COMPLETE in {elapsed_suite:.2f} seconds.")
    print(f"Raw tabular records: {config.csv_log_path}")
    print(f"Full JSONL traces:   {config.jsonl_log_path}")
    print("=" * 78)


def main() -> None:
    """Parse command-line arguments and start experiment suite execution."""
    parser = argparse.ArgumentParser(
        description="Run multi-agent prompt-injection contagion experiments."
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=default_config.provider,
        help="LLM provider identifier or custom profile."
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Custom OpenAI-compatible base URL."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Explicit provider API key."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_config.model_name,
        help="Target model identifier."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=default_config.num_runs,
        help="Number of trials per experimental cell."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a rapid 3-trial verification test."
    )
    parser.add_argument(
        "--compromise-mode",
        type=str,
        choices=["system_prompt", "input"],
        default=default_config.compromise_mode.value,
        help="Method of compromising Patient Zero."
    )
    parser.add_argument(
        "--task-type",
        type=str,
        choices=["summarization", "extraction", "translation", "json_structuring", "formal_rewrite"],
        default=default_config.downstream_task.value,
        help="Task assigned to downstream agents."
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["plain_python", "langgraph"],
        default=default_config.topology_backend.value,
        help="Topology execution backend."
    )
    parser.add_argument(
        "--chain-length",
        type=int,
        default=default_config.chain_length,
        help="Number of agent hops in the pipeline."
    )
    parser.add_argument(
        "--topologies",
        type=str,
        default="all",
        help="Comma-separated topologies to run: 'all', or a subset like 'chain,star,ring,tree'."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Inter-request delay in seconds (defaults to provider's free-tier auto-delay)."
    )

    args = parser.parse_args()
    num_runs = 3 if args.smoke_test else args.runs

    selected_topologies = None
    if args.topologies and args.topologies.lower() != "all":
        raw_list = [t.strip().lower() for t in args.topologies.split(",") if t.strip()]
        selected_topologies = [TopologyType(t) for t in raw_list]

    config = ExperimentConfig(
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=args.model,
        num_runs=num_runs,
        compromise_mode=CompromiseMode(args.compromise_mode),
        downstream_task=DownstreamTask(args.task_type),
        topology_backend=TopologyBackend(args.backend),
        chain_length=args.chain_length,
        request_delay_sec=args.delay
    )

    run_experiment_suite(config, selected_topologies=selected_topologies)


if __name__ == "__main__":
    main()
