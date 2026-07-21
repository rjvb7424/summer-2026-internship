"""
analyze_results.py
==================

Reads a run's ``results.json`` and writes plots visualising how well each model
achieved the objective.

Usage:

    python analyze_results.py
    python analyze_results.py my_config.yaml
    python analyze_results.py --results runs/gather_wood_10x10/results.json

Plots are written to ``<run_dir>/plots/``:

  * success_rate.png     - fraction of trials solved, per model
  * turns_to_success.png - turns needed when solved
  * think_time.png       - mean seconds per decision, per model
  * success_matrix.png   - model x trial grid

The experiment name is included in every graph title. Underscores in the
experiment name are replaced with spaces.

For example:

    tree_opening_9x9

becomes:

    tree opening 9x9
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

# Use a non-interactive backend so plots can be generated without opening
# Matplotlib windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Palette
# =============================================================================

# Muted, colour-blind-friendly colours.
BAR_COLOR = "#4c72b0"
THINK_TIME_COLOR = "#8172b3"
OK_COLOR = "#2f9e6f"
FAIL_COLOR = "#c44e52"
GRID_BG = "#e9e6df"
NOT_RUN_COLOR = "#c9ccd6"


# =============================================================================
# Experiment-name extraction
# =============================================================================

def get_experiment_name(
    results: dict[str, Any],
    results_path: Path | None = None,
) -> str:
    """
    Extract and format the experiment name.

    The preferred results.json structure is:

        {
            "experiment": {
                "name": "tree_opening_9x9"
            }
        }

    The function also checks alternative locations and finally falls back to
    the name of the run directory.

    Underscores are replaced with spaces.
    """

    experiment = results.get("experiment", {})

    if isinstance(experiment, dict):
        name = experiment.get("name")
    elif isinstance(experiment, str):
        name = experiment
    else:
        name = None

    # Alternative locations in case results.json uses another structure.
    name = (
        name
        or results.get("experiment_name")
        or results.get("name")
    )

    # If the name is not recorded in results.json, use the run folder name.
    if not name and results_path is not None:
        name = results_path.parent.name

    if not name:
        name = "experiment"

    return str(name).replace("_", " ").strip()


# =============================================================================
# Statistics extraction
# =============================================================================

def summarise(results: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Collapse the raw experiment transcript into one statistics row per model.
    """

    num_trials = int(results.get("num_trials", 0))
    rows: list[dict[str, Any]] = []

    models = results.get("models", {})

    for name, record in models.items():
        trials = record.get("trials", [])

        successes = [
            trial
            for trial in trials
            if bool(trial.get("success", False))
        ]

        solve_turns = [
            int(trial["success_turn"]) + 1
            for trial in successes
            if trial.get("success_turn") is not None
        ]

        think_times = [
            float(turn["think_seconds"])
            for trial in trials
            for turn in trial.get("turns", [])
            if turn.get("think_seconds") is not None
        ]

        rows.append(
            {
                "name": name,
                "error": record.get("error"),
                "n_trials": len(trials),
                "n_success": len(successes),
                "success_rate": (
                    len(successes) / len(trials)
                    if trials
                    else 0.0
                ),
                "solve_turns": solve_turns,
                "mean_solve_turns": (
                    float(np.mean(solve_turns))
                    if solve_turns
                    else None
                ),
                "mean_think": (
                    float(np.mean(think_times))
                    if think_times
                    else None
                ),
                "per_trial_success": [
                    bool(trial.get("success", False))
                    for trial in trials
                ],
                "expected_trials": num_trials,
            }
        )

    return rows


# =============================================================================
# Plots
# =============================================================================

def plot_success_rate(
    rows: list[dict[str, Any]],
    out: Path,
    experiment_name: str,
) -> None:
    """Plot the percentage of successful trials for each model."""

    names = [row["name"] for row in rows]
    values = [
        100 * row["success_rate"]
        for row in rows
    ]

    not_run = [
        bool(row["error"]) or row["n_trials"] == 0
        for row in rows
    ]

    colours = [
        NOT_RUN_COLOR if unavailable else BAR_COLOR
        for unavailable in not_run
    ]

    fig, ax = plt.subplots(
        figsize=(max(6, 1.6 * len(names)), 4.5)
    )

    ax.bar(
        names,
        values,
        color=colours,
    )

    for index, row in enumerate(rows):
        if not_run[index]:
            label = "error" if row["error"] else "not run"

            ax.text(
                index,
                2,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                color="#555555",
            )
        else:
            ax.text(
                index,
                values[index] + 1,
                f"{values[index]:.0f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title(
        f"Success rate of {experiment_name}"
    )

    _finish(fig, ax, out)


def plot_turns_to_success(
    rows: list[dict[str, Any]],
    out: Path,
    experiment_name: str,
) -> None:
    """
    Plot the mean number of turns needed to solve the objective.

    Individual successful trials are displayed as dots.
    """

    names = [row["name"] for row in rows]

    means = [
        row["mean_solve_turns"]
        if row["mean_solve_turns"] is not None
        else 0
        for row in rows
    ]

    fig, ax = plt.subplots(
        figsize=(max(6, 1.6 * len(names)), 4.5)
    )

    ax.bar(
        names,
        means,
        color=BAR_COLOR,
        alpha=0.75,
    )

    for index, row in enumerate(rows):
        solve_turns = row["solve_turns"]

        if not solve_turns:
            continue

        # Use a fixed random seed for each model so the jitter is reproducible.
        random_state = np.random.RandomState(index)

        jitter = (
            random_state.rand(len(solve_turns)) - 0.5
        ) * 0.25

        x_positions = (
            np.full(len(solve_turns), index)
            + jitter
        )

        ax.scatter(
            x_positions,
            solve_turns,
            color=OK_COLOR,
            s=28,
            zorder=3,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_ylabel("Turns to reach the goal")
    ax.set_title(
        f"Turns to success of {experiment_name}\n"
        "(bar = mean, dots = successful trials)"
    )

    _finish(fig, ax, out)


def plot_think_time(
    rows: list[dict[str, Any]],
    out: Path,
    experiment_name: str,
) -> None:
    """Plot the mean model decision time per turn."""

    names = [row["name"] for row in rows]

    means = [
        row["mean_think"]
        if row["mean_think"] is not None
        else 0
        for row in rows
    ]

    fig, ax = plt.subplots(
        figsize=(max(6, 1.6 * len(names)), 4.5)
    )

    ax.bar(
        names,
        means,
        color=THINK_TIME_COLOR,
    )

    for index, value in enumerate(means):
        if value > 0:
            ax.text(
                index,
                value,
                f"{value:.2f}s",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        else:
            ax.text(
                index,
                0,
                "-",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_ylabel("Mean think time per turn (s)")
    ax.set_title(
        f"Think time of {experiment_name}"
    )

    _finish(fig, ax, out)


def plot_success_matrix(
    rows: list[dict[str, Any]],
    out: Path,
    experiment_name: str,
) -> None:
    """
    Plot a model-by-trial grid showing successful and failed trials.

    Green cells indicate success.
    Red cells indicate failure.
    Blank cells indicate trials that were not run.
    """

    names = [row["name"] for row in rows]

    expected_trials = max(
        (
            int(row["expected_trials"])
            for row in rows
        ),
        default=0,
    )

    recorded_trials = max(
        (
            int(row["n_trials"])
            for row in rows
        ),
        default=1,
    )

    n_columns = expected_trials or recorded_trials

    grid = np.full(
        (len(names), n_columns),
        np.nan,
    )

    for model_index, row in enumerate(rows):
        for trial_index, succeeded in enumerate(
            row["per_trial_success"]
        ):
            if trial_index >= n_columns:
                break

            grid[model_index, trial_index] = (
                1.0 if succeeded else 0.0
            )

    fig, ax = plt.subplots(
        figsize=(
            max(6, 0.8 * n_columns + 3),
            max(3, 0.7 * len(names) + 2),
        )
    )

    colour_map = matplotlib.colors.ListedColormap(
        [
            FAIL_COLOR,
            OK_COLOR,
        ]
    )

    ax.set_facecolor(GRID_BG)

    masked_grid = np.ma.masked_invalid(grid)

    ax.imshow(
        masked_grid,
        cmap=colour_map,
        vmin=0,
        vmax=1,
        aspect="auto",
    )

    ax.set_xticks(
        range(n_columns),
        [
            f"T{trial_index + 1}"
            for trial_index in range(n_columns)
        ],
    )

    ax.set_yticks(
        range(len(names)),
        names,
    )

    for model_index in range(len(names)):
        for trial_index in range(n_columns):
            value = grid[model_index, trial_index]

            if np.isnan(value):
                continue

            ax.text(
                trial_index,
                model_index,
                "✓" if value == 1.0 else "✗",
                ha="center",
                va="center",
                color="white",
                fontsize=11,
            )

    ax.set_title(
        f"Success matrix of {experiment_name}\n"
        "(green = solved, red = failed, blank = not run)"
    )

    fig.tight_layout()
    fig.savefig(
        out,
        dpi=130,
        bbox_inches="tight",
    )
    plt.close(fig)


def _finish(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    out: Path,
) -> None:
    """Apply shared plot formatting and save the figure."""

    ax.spines[["top", "right"]].set_visible(False)

    plt.setp(
        ax.get_xticklabels(),
        rotation=20,
        ha="right",
    )

    fig.tight_layout()

    fig.savefig(
        out,
        dpi=130,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# Console summary
# =============================================================================

def print_summary(
    results: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    """Print a summary table of the experiment results."""

    objective = results.get("objective", {})

    if isinstance(objective, dict):
        objective_name = objective.get(
            "label",
            objective.get("target", "?"),
        )
    else:
        objective_name = str(objective)

    print(
        f"\nObjective: {objective_name}"
    )

    print(
        f"{'model':<34} "
        f"{'success':>9} "
        f"{'rate':>7} "
        f"{'avg turns':>10} "
        f"{'avg think':>10}"
    )

    print("-" * 74)

    for row in rows:
        mean_solve_turns = row["mean_solve_turns"]
        mean_think = row["mean_think"]

        turns_text = (
            f"{mean_solve_turns:.1f}"
            if mean_solve_turns is not None
            else "-"
        )

        think_text = (
            f"{mean_think:.2f}s"
            if mean_think is not None
            else "-"
        )

        error_flag = (
            "  [load error]"
            if row["error"]
            else ""
        )

        print(
            f"{row['name']:<34} "
            f"{row['n_success']:>4}/{row['n_trials']:<4} "
            f"{100 * row['success_rate']:>6.0f}% "
            f"{turns_text:>10} "
            f"{think_text:>10}"
            f"{error_flag}"
        )

    print()


# =============================================================================
# Results-path resolution
# =============================================================================

def resolve_results_path(
    args: argparse.Namespace,
) -> Path:
    """
    Resolve results.json from either --results or the experiment config.
    """

    if args.results:
        return Path(args.results)

    from config import load_config

    config = load_config(args.config)

    return config.results_path


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    """Read results.json, print statistics, and build all plots."""

    argument_parser = argparse.ArgumentParser(
        description="Plot Crafter experiment results."
    )

    argument_parser.add_argument(
        "config",
        nargs="?",
        default="config.yaml",
        help=(
            "config.yaml used for the run "
            "(default: config.yaml)"
        ),
    )

    argument_parser.add_argument(
        "--results",
        help=(
            "path to results.json; "
            "overrides the config path"
        ),
    )

    args = argument_parser.parse_args()

    results_path = resolve_results_path(args)

    results = json.loads(
        results_path.read_text(
            encoding="utf-8"
        )
    )

    rows = summarise(results)

    experiment_name = get_experiment_name(
        results,
        results_path,
    )

    plots_dir = (
        results_path.parent
        / "plots"
    )

    plots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_success_rate(
        rows,
        plots_dir / "success_rate.png",
        experiment_name,
    )

    plot_turns_to_success(
        rows,
        plots_dir / "turns_to_success.png",
        experiment_name,
    )

    plot_think_time(
        rows,
        plots_dir / "think_time.png",
        experiment_name,
    )

    plot_success_matrix(
        rows,
        plots_dir / "success_matrix.png",
        experiment_name,
    )

    print_summary(
        results,
        rows,
    )

    print(
        f"Plots written to {plots_dir}/"
    )


if __name__ == "__main__":
    main()