"""
analyze_results.py
==================

Reads a run's ``results.json`` and writes plots visualising how well each model
achieved the objective.

    python analyze_results.py                 # uses config.yaml to find the run
    python analyze_results.py my_config.yaml
    python analyze_results.py --results runs/gather_wood_10x10/results.json

Plots written to ``<run_dir>/plots/``:
  * success_rate.png     - fraction of trials solved, per model
  * turns_to_success.png - turns needed when solved (mean bar + per-trial dots)
  * think_time.png       - mean seconds per decision, per model
  * success_matrix.png   - model x trial grid (solved / failed / not run)

This is an analysis tool, so it prints a summary table to the console.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- Palette (muted, colour-blind friendly) ----------------------------------
BAR_COLOR = "#4c72b0"
OK_COLOR = "#2f9e6f"
FAIL_COLOR = "#c44e52"
GRID_BG = "#e9e6df"
NOT_RUN_COLOR = "#c9ccd6"  # model that errored or has zero recorded trials


# =============================================================================
#  Stats extraction
# =============================================================================
def summarise(results: dict) -> list[dict]:
    """Collapse the raw transcript into one stats row per model."""
    num_trials = int(results.get("num_trials", 0))
    rows: list[dict] = []
    for name, rec in results["models"].items():
        trials = rec.get("trials", [])
        successes = [t for t in trials if t["success"]]
        solve_turns = [t["success_turn"] + 1 for t in successes if t["success_turn"] is not None]
        think = [
            turn["think_seconds"]
            for t in trials for turn in t["turns"]
        ]
        rows.append({
            "name": name,
            "error": rec.get("error"),
            "n_trials": len(trials),
            "n_success": len(successes),
            "success_rate": (len(successes) / len(trials)) if trials else 0.0,
            "solve_turns": solve_turns,
            "mean_solve_turns": float(np.mean(solve_turns)) if solve_turns else None,
            "mean_think": float(np.mean(think)) if think else None,
            "per_trial_success": [bool(t["success"]) for t in trials],
            "expected_trials": num_trials,
        })
    return rows


# =============================================================================
#  Plots
# =============================================================================
def plot_success_rate(rows, out: Path) -> None:
    names = [r["name"] for r in rows]
    values = [100 * r["success_rate"] for r in rows]
    not_run = [bool(r["error"]) or r["n_trials"] == 0 for r in rows]
    colors = [NOT_RUN_COLOR if nr else BAR_COLOR for nr in not_run]
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(names)), 4.5))
    ax.bar(names, values, color=colors)
    for i, r in enumerate(rows):
        if not_run[i]:
            ax.text(i, 2, "error" if r["error"] else "not run",
                    ha="center", va="bottom", fontsize=9, color="#555")
        else:
            ax.text(i, values[i] + 1, f"{values[i]:.0f}%",
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Objective success rate by model")
    _finish(fig, ax, out)


def plot_turns_to_success(rows, out: Path) -> None:
    names = [r["name"] for r in rows]
    means = [r["mean_solve_turns"] or 0 for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(names)), 4.5))
    ax.bar(names, means, color=BAR_COLOR, alpha=0.75)
    for i, r in enumerate(rows):
        if r["solve_turns"]:
            jitter = (np.random.RandomState(i).rand(len(r["solve_turns"])) - 0.5) * 0.25
            ax.scatter(np.full(len(r["solve_turns"]), i) + jitter, r["solve_turns"],
                       color=OK_COLOR, s=28, zorder=3, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Turns to reach the goal")
    ax.set_title("Turns needed when solved (bar = mean, dots = trials)")
    _finish(fig, ax, out)


def plot_think_time(rows, out: Path) -> None:
    names = [r["name"] for r in rows]
    means = [r["mean_think"] or 0 for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(names)), 4.5))
    ax.bar(names, means, color="#8172b3")
    for i, v in enumerate(means):
        ax.text(i, v, f"{v:.2f}s", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean think time per turn (s)")
    ax.set_title("Average decision latency by model")
    _finish(fig, ax, out)


def plot_success_matrix(rows, out: Path) -> None:
    names = [r["name"] for r in rows]
    n_cols = max((r["expected_trials"] for r in rows), default=0) or \
        max((r["n_trials"] for r in rows), default=1)
    grid = np.full((len(names), n_cols), np.nan)
    for i, r in enumerate(rows):
        for j, ok in enumerate(r["per_trial_success"]):
            grid[i, j] = 1.0 if ok else 0.0

    fig, ax = plt.subplots(figsize=(max(6, 0.8 * n_cols + 3), 0.7 * len(names) + 2))
    cmap = matplotlib.colors.ListedColormap([FAIL_COLOR, OK_COLOR])
    ax.set_facecolor(GRID_BG)
    ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_cols), [f"T{j + 1}" for j in range(n_cols)])
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(n_cols):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, "✓" if grid[i, j] else "✗", ha="center", va="center",
                        color="white", fontsize=11)
    ax.set_title("Per-trial outcomes  (green = solved, red = failed, blank = not run)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _finish(fig, ax, out: Path) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# =============================================================================
#  Console summary + entry point
# =============================================================================
def print_summary(results: dict, rows) -> None:
    obj = results.get("objective", {})
    print(f"\nObjective: {obj.get('label', obj.get('target', '?'))}")
    print(f"{'model':<34} {'success':>9} {'rate':>7} {'avg turns':>10} {'avg think':>10}")
    print("-" * 74)
    for r in rows:
        turns = f"{r['mean_solve_turns']:.1f}" if r["mean_solve_turns"] else "-"
        think = f"{r['mean_think']:.2f}s" if r["mean_think"] else "-"
        flag = "  [load error]" if r["error"] else ""
        print(f"{r['name']:<34} {r['n_success']:>4}/{r['n_trials']:<4} "
              f"{100 * r['success_rate']:>6.0f}% {turns:>10} {think:>10}{flag}")
    print()


def resolve_results_path(args) -> Path:
    if args.results:
        return Path(args.results)
    from config import load_config
    cfg = load_config(args.config)
    return cfg.results_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Crafter experiment results.")
    ap.add_argument("config", nargs="?", default="config.yaml",
                    help="config.yaml used for the run (to locate results.json)")
    ap.add_argument("--results", help="path to results.json (overrides config)")
    args = ap.parse_args()

    results_path = resolve_results_path(args)
    results = json.loads(Path(results_path).read_text())
    rows = summarise(results)

    plots_dir = results_path.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_success_rate(rows, plots_dir / "success_rate.png")
    plot_turns_to_success(rows, plots_dir / "turns_to_success.png")
    plot_think_time(rows, plots_dir / "think_time.png")
    plot_success_matrix(rows, plots_dir / "success_matrix.png")

    print_summary(results, rows)
    print(f"Plots written to {plots_dir}/")


if __name__ == "__main__":
    main()
