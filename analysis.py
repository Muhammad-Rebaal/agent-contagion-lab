"""Statistical analysis engine computing Wilson score confidence intervals and Fisher's exact tests."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from scipy import stats

from config import default_config


def resolve_target_csv_path(target_path: Optional[Path] = None) -> Path:
    """Resolve the target CSV file path from explicit arguments, latest pointers, or timestamped outputs."""
    if target_path and target_path.exists():
        return target_path

    output_dir = default_config.output_dir
    if output_dir.exists():
        latest_file = output_dir / "output_latest.csv"
        if latest_file.exists():
            return latest_file
        csv_files = sorted(output_dir.glob("output_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if csv_files:
            return csv_files[0]

    legacy = default_config.base_dir / "results" / "raw_experiment_data.csv"
    if legacy.exists():
        return legacy

    return default_config.csv_log_path


@dataclass
class ProportionStats:
    """Statistical summary for an observed binomial proportion including Wilson confidence intervals."""
    k_successes: int
    n_trials: int
    proportion: float
    ci_lower: float
    ci_upper: float

    def to_pct_str(self) -> str:
        """Format the proportion and 95% confidence interval as a human-readable percentage string."""
        return f"{self.proportion * 100:.1f}% [{self.ci_lower * 100:.1f}%, {self.ci_upper * 100:.1f}%]"


def compute_wilson_score_interval(
    k: int,
    n: int,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """Calculate the Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0

    p_hat = k / n
    alpha = 1.0 - confidence
    z = stats.norm.ppf(1.0 - alpha / 2.0)

    denominator = n + z ** 2
    center = (k + (z ** 2) / 2.0) / denominator
    spread = (z / denominator) * math.sqrt(
        (k * (n - k) / n) + (z ** 2) / 4.0
    )

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)

    return round(p_hat, 4), round(lower, 4), round(upper, 4)


def calculate_proportion_stats(k: int, n: int) -> ProportionStats:
    """Wrap Wilson score interval calculation in a structured ProportionStats object."""
    p, lower, upper = compute_wilson_score_interval(k, n)
    return ProportionStats(
        k_successes=k,
        n_trials=n,
        proportion=p,
        ci_lower=lower,
        ci_upper=upper
    )


def analyze_experiment_results(
    csv_path: Optional[Path] = None,
    output_json_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Analyze experiment telemetry and calculate infection rates, significance tests, and decay curves."""
    target_csv = resolve_target_csv_path(csv_path)
    if not target_csv.exists():
        raise FileNotFoundError(
            f"Experiment data not found at '{target_csv}'. Run experiments first via runner.py."
        )

    print(f"Loading experiment dataset: {target_csv}")
    df = pd.read_csv(target_csv)

    for col in ["exact_match", "near_miss", "is_infected"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    metrics: Dict[str, Any] = {
        "metadata": {
            "total_rows": len(df),
            "unique_runs": df["run_id"].nunique(),
            "models_tested": df["model"].unique().tolist(),
            "topologies_tested": df["topology"].unique().tolist(),
            "conditions_tested": df["condition"].unique().tolist(),
        }
    }

    df_control = df[df["condition"] == "control"]
    if not df_control.empty:
        total_control_hops = len(df_control)
        false_positives = int(df_control["is_infected"].sum())
        fp_rate, fp_low, fp_high = compute_wilson_score_interval(false_positives, total_control_hops)
        metrics["control_audit"] = {
            "total_control_agent_hops": total_control_hops,
            "false_positive_count": false_positives,
            "false_positive_rate": fp_rate,
            "ci_95": [fp_low, fp_high],
            "passed_audit": bool(false_positives == 0)
        }
    else:
        metrics["control_audit"] = {"status": "No control condition runs found."}

    df_attack = df[df["condition"] == "attack"]
    topology_summaries: Dict[str, Any] = {}

    for topology in df_attack["topology"].unique():
        df_top = df_attack[df_attack["topology"] == topology]
        unique_runs = df_top["run_id"].unique()
        n_trials = len(unique_runs)

        pz_infected_run_ids = set(
            df_top[(df_top["hop_index"] == 0) & (df_top["is_infected"] == True)]["run_id"]
        )
        pz_k = len(pz_infected_run_ids)
        pz_stats = calculate_proportion_stats(pz_k, n_trials)

        agent_stats: Dict[str, Any] = {}

        for agent_name in sorted(df_top["agent_name"].unique()):
            df_ag = df_top[df_top["agent_name"] == agent_name]

            k_uncond = int(df_ag["is_infected"].sum())
            uncond_stats = calculate_proportion_stats(k_uncond, n_trials)

            if pz_k > 0:
                df_cond = df_ag[df_ag["run_id"].isin(pz_infected_run_ids)]
                k_cond = int(df_cond["is_infected"].sum())
                cond_stats = calculate_proportion_stats(k_cond, pz_k)
            else:
                cond_stats = ProportionStats(0, 0, 0.0, 0.0, 0.0)

            exact_count = int(df_ag["exact_match"].sum())
            near_miss_count = int(df_ag["near_miss"].sum())

            agent_stats[agent_name] = {
                "unconditional": asdict(uncond_stats),
                "conditional": asdict(cond_stats),
                "exact_matches": exact_count,
                "near_misses": near_miss_count
            }

        topology_summaries[topology] = {
            "n_runs": n_trials,
            "patient_zero_activation": asdict(pz_stats),
            "agents": agent_stats
        }

    metrics["attack_summary"] = topology_summaries

    hypothesis_results: Dict[str, Any] = {}

    if "chain" in topology_summaries and "star" in topology_summaries:
        chain_c = topology_summaries["chain"]["agents"].get("Agent_C", {}).get("unconditional")
        star_c = topology_summaries["star"]["agents"].get("Agent_C", {}).get("unconditional")

        if chain_c and star_c:
            k_chain = chain_c["k_successes"]
            n_chain = chain_c["n_trials"]
            k_star = star_c["k_successes"]
            n_star = star_c["n_trials"]

            contingency_table = [
                [k_star, n_star - k_star],
                [k_chain, n_chain - k_chain]
            ]

            odds_ratio, p_value = stats.fisher_exact(contingency_table, alternative="two-sided")
            diff_prop = star_c["proportion"] - chain_c["proportion"]

            hypothesis_results["star_vs_chain"] = {
                "comparison": "Agent C in Star (direct) vs Agent C in Chain (summarized)",
                "star_agent_c_rate": star_c["proportion"],
                "chain_agent_c_rate": chain_c["proportion"],
                "difference_in_proportions": round(diff_prop, 4),
                "contingency_table": contingency_table,
                "odds_ratio": round(float(odds_ratio), 4) if not math.isinf(odds_ratio) else "inf",
                "fisher_exact_p_value": round(float(p_value), 5),
                "is_statistically_significant_05": bool(p_value < 0.05)
            }

    if "ring" in topology_summaries:
        ring_loop = topology_summaries["ring"]["agents"].get("Agent_A_loop", {}).get("unconditional")
        if ring_loop:
            hypothesis_results["ring_loop_recurrence"] = {
                "comparison": "Ring feedback closure (Agent A loop re-infection rate)",
                "loop_infection_rate": ring_loop["proportion"],
                "ci_95": [ring_loop["ci_lower"], ring_loop["ci_upper"]],
                "persisted_through_cycle": bool(ring_loop["k_successes"] > 0)
            }

    if "tree" in topology_summaries:
        tree_agents = topology_summaries["tree"]["agents"]
        leaves_infected = sum(
            tree_agents.get(name, {}).get("unconditional", {}).get("k_successes", 0)
            for name in ["Agent_D", "Agent_E"]
        )
        leaves_total = sum(
            tree_agents.get(name, {}).get("unconditional", {}).get("n_trials", 0)
            for name in ["Agent_D", "Agent_E"]
        )
        if leaves_total > 0:
            l_stat = calculate_proportion_stats(leaves_infected, leaves_total)
            hypothesis_results["tree_leaf_penetration"] = {
                "comparison": "Tree hierarchical leaves (Agent D & E aggregate infection)",
                "leaf_infection_rate": l_stat.proportion,
                "ci_95": [l_stat.ci_lower, l_stat.ci_upper],
                "reached_leaves": bool(leaves_infected > 0)
            }

    metrics["hypothesis_testing"] = hypothesis_results

    df_chain_attack = df_attack[df_attack["topology"] == "chain"]
    if not df_chain_attack.empty:
        decay_curve: List[Dict[str, Any]] = []
        for hop in sorted(df_chain_attack["hop_index"].unique()):
            df_hop = df_chain_attack[df_chain_attack["hop_index"] == hop]
            k_hop = int(df_hop["is_infected"].sum())
            n_hop = len(df_hop)
            hop_stats = calculate_proportion_stats(k_hop, n_hop)
            decay_curve.append({
                "hop_index": int(hop),
                "agent_name": df_hop["agent_name"].iloc[0],
                "stats": asdict(hop_stats)
            })
        metrics["chain_hop_decay"] = decay_curve

    save_path = output_json_path or default_config.summary_json_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def print_formatted_report(metrics: Dict[str, Any]) -> None:
    """Print a structured summary of statistical findings and hypothesis tests to the console."""
    print("\n" + "=" * 78)
    print("CONTAGION LAB: SCIENTIFIC RESULTS & STATISTICAL ANALYSIS")
    print("=" * 78)

    ca = metrics.get("control_audit", {})
    if "false_positive_rate" in ca:
        fp_status = "PASSED (0.0% FP)" if ca["passed_audit"] else f"WARNING: {ca['false_positive_count']} FPs"
        print(f"Control Condition Audit:  {fp_status} across {ca['total_control_agent_hops']} agent hops.")
    print("-" * 78)

    attack_data = metrics.get("attack_summary", {})
    for top_name, top_val in attack_data.items():
        pz = top_val["patient_zero_activation"]
        print(f"\nTopology: {top_name.upper()} (N = {top_val['n_runs']} runs)")
        print(f"  Patient Zero (Agent A) Success: {pz['proportion']*100:.1f}% "
              f"[{pz['ci_lower']*100:.1f}%, {pz['ci_upper']*100:.1f}%]")

        print("  Downstream Agent Infection Rates:")
        for ag_name, ag_data in top_val["agents"].items():
            u = ag_data["unconditional"]
            c = ag_data["conditional"]
            exact = ag_data["exact_matches"]
            near = ag_data["near_misses"]
            print(f"    - {ag_name:12s} | Unconditional: {u['proportion']*100:5.1f}% [{u['ci_lower']*100:4.1f}%, {u['ci_upper']*100:4.1f}%] "
                  f"| Given Patient Zero: {c['proportion']*100:5.1f}% [{c['ci_lower']*100:4.1f}%, {c['ci_upper']*100:4.1f}%] "
                  f"| (Exact: {exact}, Near-miss: {near})")

    ht = metrics.get("hypothesis_testing", {})
    if ht:
        print("\n" + "-" * 78)
        print("TOPOLOGY HYPOTHESIS TESTING & COMPARISONS:")
        if "star_vs_chain" in ht:
            svc = ht["star_vs_chain"]
            print(f"  [Star vs Chain] {svc['comparison']}")
            print(f"    Star Agent C:  {svc['star_agent_c_rate']*100:.1f}%")
            print(f"    Chain Agent C: {svc['chain_agent_c_rate']*100:.1f}%")
            print(f"    Difference:    {svc['difference_in_proportions']*100:+.1f} percentage points")
            print(f"    Odds Ratio:    {svc['odds_ratio']}")
            print(f"    Fisher's p:    {svc['fisher_exact_p_value']} (Sig: {'YES' if svc['is_statistically_significant_05'] else 'NO'})")
        if "ring_loop_recurrence" in ht:
            rl = ht["ring_loop_recurrence"]
            print(f"  [Ring Feedback] Loop Re-infection: {rl['loop_infection_rate']*100:.1f}% [CI: {rl['ci_95'][0]*100:.1f}%, {rl['ci_95'][1]*100:.1f}%]")
        if "tree_leaf_penetration" in ht:
            tl = ht["tree_leaf_penetration"]
            print(f"  [Tree Hierarchy] Leaf Penetration: {tl['leaf_infection_rate']*100:.1f}% [CI: {tl['ci_95'][0]*100:.1f}%, {tl['ci_95'][1]*100:.1f}%]")
        print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze multi-agent contagion experiment results.")
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Path to specific CSV file in output/ (defaults to most recent run)."
    )
    args = parser.parse_args()
    target_file = Path(args.file) if args.file else None
    results = analyze_experiment_results(csv_path=target_file)
    print_formatted_report(results)
