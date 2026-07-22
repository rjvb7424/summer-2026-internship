"""
analyze_results.py
==================

Reads a run's results.json and writes plots visualising how well each model
achieved the objective. The experiment name appears in every title (underscores
shown as spaces). Plots are written to <run_dir>/plots/:

  * success_rate.png      - fraction of trials solved, per model
  * turns_to_success.png  - turns needed when solved
  * think_time.png        - mean seconds per decision, per model
  * success_matrix.png    - model x trial grid
  * token_usage.png       - tokens vs turns per trial, coloured by model,
                            circle = solved, x = failed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

BAR_COLOR = "#4c72b0"
THINK_TIME_COLOR = "#8172b3"
OK_COLOR = "#2f9e6f"
FAIL_COLOR = "#c44e52"
GRID_BG = "#e9e6df"
NOT_RUN_COLOR = "#c9ccd6"


def get_experiment_name(results: dict[str, Any], results_path: Path | None = None) -> str:
    """Extract the experiment name and replace underscores with spaces."""
    experiment = results.get("experiment", {})
    if isinstance(experiment, dict):
        name = experiment.get("name")
    elif isinstance(experiment, str):
        name = experiment
    else:
        name = None
    name = name or results.get("experiment_name") or results.get("name")
    if not name and results_path is not None:
        name = results_path.parent.name
    if not name:
        name = "experiment"
    return str(name).replace("_", " ").strip()


def summarise(results: dict[str, Any]) -> list[dict[str, Any]]:
    """One statistics row per model."""
    num_trials = int(results.get("num_trials", 0))
    rows: list[dict[str, Any]] = []
    for name, record in results.get("models", {}).items():
        trials = record.get("trials", [])
        successes = [t for t in trials if bool(t.get("success", False))]
        solve_turns = [
            int(t["success_turn"]) + 1 for t in successes
            if t.get("success_turn") is not None
        ]
        think_times = [
            float(turn["think_seconds"]) for t in trials
            for turn in t.get("turns", []) if turn.get("think_seconds") is not None
        ]
        rows.append({
            "name": name,
            "error": record.get("error"),
            "n_trials": len(trials),
            "n_success": len(successes),
            "success_rate": len(successes) / len(trials) if trials else 0.0,
            "solve_turns": solve_turns,
            "mean_solve_turns": float(np.mean(solve_turns)) if solve_turns else None,
            "mean_think": float(np.mean(think_times)) if think_times else None,
            "per_trial_success": [bool(t.get("success", False)) for t in trials],
            "expected_trials": num_trials,
        })
    return rows


def plot_success_rate(rows, out, experiment_name):
    names = [r["name"] for r in rows]
    values = [100 * r["success_rate"] for r in rows]
    not_run = [bool(r["error"]) or r["n_trials"] == 0 for r in rows]
    colours = [NOT_RUN_COLOR if nr else BAR_COLOR for nr in not_run]
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(names)), 4.5))
    ax.bar(names, values, color=colours)
    for i, r in enumerate(rows):
        if not_run[i]:
            ax.text(i, 2, "error" if r["error"] else "not run",
                    ha="center", va="bottom", fontsize=9, color="#555555")
        else:
            ax.text(i, values[i] + 1, f"{values[i]:.0f}%",
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"Success rate of {experiment_name}")
    _finish(fig, ax, out)


def plot_turns_to_success(rows, out, experiment_name):
    names = [r["name"] for r in rows]
    means = [r["mean_solve_turns"] if r["mean_solve_turns"] is not None else 0 for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(names)), 4.5))
    ax.bar(names, means, color=BAR_COLOR, alpha=0.75)
    for i, r in enumerate(rows):
        st = r["solve_turns"]
        if not st:
            continue
        rs = np.random.RandomState(i)
        jitter = (rs.rand(len(st)) - 0.5) * 0.25
        ax.scatter(np.full(len(st), i) + jitter, st, color=OK_COLOR, s=28,
                   zorder=3, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Turns to reach the goal")
    ax.set_title(f"Turns to success of {experiment_name}\n(bar = mean, dots = successful trials)")
    _finish(fig, ax, out)


def plot_think_time(rows, out, experiment_name):
    names = [r["name"] for r in rows]
    means = [r["mean_think"] if r["mean_think"] is not None else 0 for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(names)), 4.5))
    ax.bar(names, means, color=THINK_TIME_COLOR)
    for i, v in enumerate(means):
        ax.text(i, v, f"{v:.2f}s" if v > 0 else "-", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean think time per turn (s)")
    ax.set_title(f"Think time of {experiment_name}")
    _finish(fig, ax, out)


def plot_success_matrix(rows, out, experiment_name):
    names = [r["name"] for r in rows]
    expected = max((int(r["expected_trials"]) for r in rows), default=0)
    recorded = max((int(r["n_trials"]) for r in rows), default=1)
    n_cols = expected or recorded
    grid = np.full((len(names), n_cols), np.nan)
    for mi, r in enumerate(rows):
        for ti, ok in enumerate(r["per_trial_success"]):
            if ti >= n_cols:
                break
            grid[mi, ti] = 1.0 if ok else 0.0
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * n_cols + 3), max(3, 0.7 * len(names) + 2)))
    cmap = matplotlib.colors.ListedColormap([FAIL_COLOR, OK_COLOR])
    ax.set_facecolor(GRID_BG)
    ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_cols), [f"T{i + 1}" for i in range(n_cols)])
    ax.set_yticks(range(len(names)), names)
    for mi in range(len(names)):
        for ti in range(n_cols):
            v = grid[mi, ti]
            if np.isnan(v):
                continue
            ax.text(ti, mi, "\u2713" if v == 1.0 else "\u2717",
                    ha="center", va="center", color="white", fontsize=11)
    ax.set_title(f"Success matrix of {experiment_name}\n(green = solved, red = failed, blank = not run)")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def collect_token_points(results):
    """One point per trial: model, turns_used, total_tokens, success.
    Returns (points, any_tokens); any_tokens is False for runs with no token data."""
    points = []
    any_tokens = False
    for name, record in results.get("models", {}).items():
        for trial in record.get("trials", []):
            tt = [int(t["tokens"]) for t in trial.get("turns", []) if t.get("tokens") is not None]
            if not tt:
                continue
            any_tokens = True
            points.append({
                "model": name,
                "turns": len(trial.get("turns", [])),
                "tokens": sum(tt),
                "success": bool(trial.get("success", False)),
            })
    return points, any_tokens


def plot_tokens_vs_turns(results, out, experiment_name):
    """Scatter: x = turns in a trial, y = total tokens that trial used.
    Colour = model; circle = solved, x = failed."""
    points, any_tokens = collect_token_points(results)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    if not any_tokens:
        ax.text(0.5, 0.5,
                "No token data in this run.\n"
                "Token usage is recorded from now on - run again to populate.",
                ha="center", va="center", fontsize=11, color="#555555")
        ax.set_axis_off()
        ax.set_title(f"Token usage of {experiment_name}")
        fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
        return
    models = sorted({p["model"] for p in points})
    cmap = plt.get_cmap("tab10" if len(models) <= 10 else "tab20")
    colour_of = {m: cmap(i % cmap.N) for i, m in enumerate(models)}
    for model in models:
        for success, marker in ((True, "o"), (False, "X")):
            xs = [p["turns"] for p in points if p["model"] == model and p["success"] == success]
            ys = [p["tokens"] for p in points if p["model"] == model and p["success"] == success]
            if not xs:
                continue
            ax.scatter(xs, ys, marker=marker, s=70, color=colour_of[model],
                       edgecolor="white", linewidth=0.6, alpha=0.9,
                       label=model if success else None, zorder=3)
    ax.set_xlabel("Turns taken in the trial")
    ax.set_ylabel("Total tokens used in the trial")
    ax.set_title(f"Token usage of {experiment_name}\n(colour = model, circle = solved, x = failed)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="model", fontsize=8, loc="best", framealpha=0.9)
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def _finish(fig, ax, out):
    ax.spines[["top", "right"]].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def print_summary(results, rows):
    objective = results.get("objective", {})
    if isinstance(objective, dict):
        objective_name = objective.get("label", objective.get("target", "?"))
    else:
        objective_name = str(objective)
    print(f"\nObjective: {objective_name}")
    print(f"{'model':<34} {'success':>9} {'rate':>7} {'avg turns':>10} {'avg think':>10}")
    print("-" * 74)
    for r in rows:
        turns_text = f"{r['mean_solve_turns']:.1f}" if r["mean_solve_turns"] is not None else "-"
        think_text = f"{r['mean_think']:.2f}s" if r["mean_think"] is not None else "-"
        flag = "  [load error]" if r["error"] else ""
        print(f"{r['name']:<34} {r['n_success']:>4}/{r['n_trials']:<4} "
              f"{100 * r['success_rate']:>6.0f}% {turns_text:>10} {think_text:>10}{flag}")
    print()


def resolve_results_path(args):
    if args.results:
        return Path(args.results)
    from config import load_config
    return load_config(args.config).results_path


def main():
    ap = argparse.ArgumentParser(description="Plot Crafter experiment results.")
    ap.add_argument("config", nargs="?", default="config.yaml")
    ap.add_argument("--results", help="path to results.json; overrides the config path")
    args = ap.parse_args()

    results_path = resolve_results_path(args)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    rows = summarise(results)
    experiment_name = get_experiment_name(results, results_path)

    plots_dir = results_path.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_success_rate(rows, plots_dir / "success_rate.png", experiment_name)
    plot_turns_to_success(rows, plots_dir / "turns_to_success.png", experiment_name)
    plot_think_time(rows, plots_dir / "think_time.png", experiment_name)
    plot_success_matrix(rows, plots_dir / "success_matrix.png", experiment_name)
    plot_tokens_vs_turns(results, plots_dir / "token_usage.png", experiment_name)

    print_summary(results, rows)
    print(f"Plots written to {plots_dir}/")


if __name__ == "__main__":
    main()
