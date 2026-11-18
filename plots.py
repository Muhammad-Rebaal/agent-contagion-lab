"""Publication-quality visualization suite for multi-agent contagion experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import compute_wilson_score_interval, resolve_target_csv_path
from config import default_config

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2,
})


def _add_watermark_if_mock(fig: plt.Figure, df: pd.DataFrame) -> None:
    """Render a semi-transparent watermark if the dataset was generated using simulated mock models."""
    is_mock = df["model"].str.contains("mock", case=False).any()
    if is_mock:
        fig.text(
            0.5, 0.5,
            "SIMULATED DATA (MockLLM)",
            fontsize=28,
            color="gray",
            alpha=0.18,
            ha="center",
            va="center",
            rotation=30,
            weight="bold"
        )


def plot_topology_comparison(
    csv_path: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> Path:
    """Generate bar chart comparing infection rates across topologies (Chain, Star, Ring, Tree) with Wilson error bars."""
    target_csv = resolve_target_csv_path(csv_path)
    save_file = output_path or (default_config.plots_dir / "topology_comparison.png")
    save_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(target_csv)
    df_attack = df[df["condition"] == "attack"]

    topologies = [t for t in ["chain", "star", "ring", "tree"] if t in df_attack["topology"].unique()]
    if not topologies:
        topologies = df_attack["topology"].unique().tolist()

    agents = ["Agent_A", "Agent_B", "Agent_C"]
    x = np.arange(len(agents))
    n_tops = len(topologies)
    width = 0.75 / max(n_tops, 1)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    colors = {
        "chain": "#1f77b4",
        "star": "#ff7f0e",
        "ring": "#2ca02c",
        "tree": "#9467bd"
    }
    labels = {
        "chain": "Chain (A -> B -> C)",
        "star": "Star (Fan-out)",
        "ring": "Ring (Feedback Loop)",
        "tree": "Tree (Hierarchical)"
    }

    for i, top in enumerate(topologies):
        df_top = df_attack[df_attack["topology"] == top]
        props = []
        err_lows = []
        err_highs = []

        for ag in agents:
            df_ag = df_top[df_top["agent_name"] == ag]
            if len(df_ag) > 0:
                k = int(df_ag["is_infected"].sum())
                n = len(df_ag)
                p, low, high = compute_wilson_score_interval(k, n)
            else:
                p, low, high = 0.0, 0.0, 0.0
            props.append(p * 100)
            err_lows.append((p - low) * 100)
            err_highs.append((high - p) * 100)

        offset = (i - 0.5 * (n_tops - 1)) * width
        ax.bar(
            x + offset,
            props,
            width,
            yerr=[err_lows, err_highs],
            capsize=4,
            label=labels.get(top, top.title()),
            color=colors.get(top, None),
            alpha=0.88,
            edgecolor="black",
            linewidth=0.8
        )

    ax.set_ylabel("Infection Rate (%)")
    ax.set_title("Canary Propagation Rate Across Network Topologies (Wilson 95% CI)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Agent A\n(Patient Zero)", "Agent B\n(Hop 1)", "Agent C\n(Hop 2 / Satellite)"])
    ax.set_ylim(0, 105)
    ax.legend(frameon=True, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    _add_watermark_if_mock(fig, df)
    plt.tight_layout()
    plt.savefig(save_file)
    plt.close()

    return save_file


def plot_topology_overview(
    csv_path: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> Path:
    """Generate executive overview bar chart comparing overall downstream contagion penetration per topology."""
    target_csv = resolve_target_csv_path(csv_path)
    save_file = output_path or (default_config.plots_dir / "topology_network_penetration.png")
    save_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(target_csv)
    df_attack = df[df["condition"] == "attack"]

    topologies = [t for t in ["chain", "star", "ring", "tree"] if t in df_attack["topology"].unique()]
    if not topologies:
        topologies = df_attack["topology"].unique().tolist()

    top_names = []
    props = []
    err_lows = []
    err_highs = []

    for top in topologies:
        df_top_downstream = df_attack[(df_attack["topology"] == top) & (df_attack["hop_index"] > 0)]
        k = int(df_top_downstream["is_infected"].sum())
        n = len(df_top_downstream)
        p, low, high = compute_wilson_score_interval(k, n)
        top_names.append(top.upper())
        props.append(p * 100)
        err_lows.append((p - low) * 100)
        err_highs.append((high - p) * 100)

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=300)
    bar_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"][:len(topologies)]

    ax.bar(
        top_names,
        props,
        yerr=[err_lows, err_highs],
        capsize=6,
        color=bar_colors,
        alpha=0.88,
        edgecolor="black",
        linewidth=0.8,
        width=0.5
    )

    ax.set_ylabel("Downstream Infection Penetration (%)")
    ax.set_title("Overall Downstream Network Vulnerability by Topology")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    _add_watermark_if_mock(fig, df)
    plt.tight_layout()
    plt.savefig(save_file)
    plt.close()

    return save_file


def plot_hop_decay(
    csv_path: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> Path:
    """Generate line chart illustrating infection rate decay across sequential hops in Chain topology."""
    target_csv = resolve_target_csv_path(csv_path)
    save_file = output_path or (default_config.plots_dir / "hop_decay_curve.png")
    save_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(target_csv)
    df_chain = df[(df["condition"] == "attack") & (df["topology"] == "chain")]

    hops = sorted(df_chain["hop_index"].unique())
    props = []
    lowers = []
    uppers = []
    hop_labels = []

    for h in hops:
        df_h = df_chain[df_chain["hop_index"] == h]
        k = int(df_h["is_infected"].sum())
        n = len(df_h)
        p, low, high = compute_wilson_score_interval(k, n)
        props.append(p * 100)
        lowers.append(low * 100)
        uppers.append(high * 100)
        ag_name = df_h["agent_name"].iloc[0]
        hop_labels.append(f"Hop {h}\n({ag_name})")

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=300)

    ax.plot(hops, props, marker="o", markersize=8, color="#d62728", linewidth=2.5, label="Observed Transmission")
    ax.fill_between(hops, lowers, uppers, color="#d62728", alpha=0.18, label="Wilson 95% Confidence Band")

    ax.set_xlabel("Transmission Distance (Hops from Patient Zero)")
    ax.set_ylabel("Infection Probability (%)")
    ax.set_title("Canary Contagion Decay Curve in Sequential Chain")
    ax.set_xticks(hops)
    ax.set_xticklabels(hop_labels)
    ax.set_ylim(-5, 105)
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.7)

    _add_watermark_if_mock(fig, df)
    plt.tight_layout()
    plt.savefig(save_file)
    plt.close()

    return save_file


def plot_detection_breakdown(
    csv_path: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> Path:
    """Generate stacked bar chart comparing exact canary matches against near-miss mutations across agents."""
    target_csv = resolve_target_csv_path(csv_path)
    save_file = output_path or (default_config.plots_dir / "detection_breakdown.png")
    save_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(target_csv)
    df_attack = df[df["condition"] == "attack"]

    agents = sorted(df_attack["agent_name"].unique())
    exact_counts = []
    near_counts = []
    clean_counts = []

    for ag in agents:
        df_ag = df_attack[df_attack["agent_name"] == ag]
        exact = int(df_ag["exact_match"].sum())
        near = int(df_ag["near_miss"].sum())
        clean = len(df_ag) - (exact + near)

        exact_counts.append(exact)
        near_counts.append(near)
        clean_counts.append(max(0, clean))

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    x = np.arange(len(agents))
    width = 0.55

    ax.bar(x, exact_counts, width, label="Exact Match", color="#2ca02c", edgecolor="black", linewidth=0.6)
    ax.bar(x, near_counts, width, bottom=exact_counts, label="Near-Miss Mutation", color="#ffbb78", edgecolor="black", linewidth=0.6)
    bottom_clean = np.array(exact_counts) + np.array(near_counts)
    ax.bar(x, clean_counts, width, bottom=bottom_clean, label="Uninfected (Clean)", color="#aec7e8", edgecolor="black", linewidth=0.6)

    ax.set_ylabel("Agent Output Count")
    ax.set_title("Canary Persistence: Exact vs Near-Miss Mutations Across Agents")
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.legend(frameon=True, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    _add_watermark_if_mock(fig, df)
    plt.tight_layout()
    plt.savefig(save_file)
    plt.close()

    return save_file


def generate_all_plots(csv_path: Optional[Path] = None) -> List[Path]:
    """Generate all experimental figures and return paths to saved image artifacts."""
    target_csv = resolve_target_csv_path(csv_path)
    if not target_csv.exists():
        raise FileNotFoundError(f"Cannot generate plots. CSV '{target_csv}' does not exist.")

    print("Generating experimental visualizations...")
    p1 = plot_topology_comparison(target_csv)
    p2 = plot_topology_overview(target_csv)
    p3 = plot_hop_decay(target_csv)
    p4 = plot_detection_breakdown(target_csv)

    print(f"  Saved: {p1}")
    print(f"  Saved: {p2}")
    print(f"  Saved: {p3}")
    print(f"  Saved: {p4}")

    return [p1, p2, p3, p4]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate publication plots from contagion experiment data.")
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Path to specific CSV file in output/ (defaults to most recent run)."
    )
    args = parser.parse_args()
    target_file = Path(args.file) if args.file else None
    generate_all_plots(csv_path=target_file)
