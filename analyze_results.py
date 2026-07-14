"""Analysis: aggregates results/*.json into comparison graphs.

Usage: python analyze_results.py  (also run automatically by main.py)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from results_store import ResultsStore


# ============================================================
# Helpers
# ============================================================
def _short(model_name):
    return model_name.split("/")[-1]


def _bar_chart(labels, means, stds, title, ylabel, filename):
    figure, axis = plt.subplots(figsize=(max(6, 1.6 * len(labels)), 4))
    positions = np.arange(len(labels))
    axis.bar(positions, means, yerr=stds, capsize=4, color="#4C8BF5")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    figure.tight_layout()
    figure.savefig(Path(config.GRAPHS_DIR) / filename, dpi=150)
    plt.close(figure)


def _metric(trials, key):
    values = np.array([trial[key] for trial in trials], dtype=float)
    return values.mean(), values.std()


# ============================================================
# Graphs
# ============================================================
def make_graphs(results):
    Path(config.GRAPHS_DIR).mkdir(parents=True, exist_ok=True)
    labels = [_short(model) for model in results]

    for key, title, ylabel, filename in [
        ("total_reward", "Mean total reward per trial", "reward", "reward.png"),
        ("steps", "Mean survival time", "steps", "survival.png"),
        ("achievements_unlocked", "Mean achievements unlocked", "achievements", "achievements.png"),
    ]:
        stats = [_metric(trials, key) for trials in results.values()]
        _bar_chart(labels, [s[0] for s in stats], [s[1] for s in stats], title, ylabel, filename)

    # Invalid action rate (percent of steps with an unparseable response).
    rates = []
    for trials in results.values():
        per_trial = [100 * trial["invalid_actions"] / trial["steps"] for trial in trials]
        rates.append((np.mean(per_trial), np.std(per_trial)))
    _bar_chart(
        labels, [r[0] for r in rates], [r[1] for r in rates],
        "Invalid action rate", "% of steps", "invalid_actions.png",
    )

    # Achievement unlock rate heatmap (share of trials unlocking each achievement).
    achievement_names = sorted(
        {name for trials in results.values() for trial in trials for name in trial["achievements"]}
    )
    if achievement_names:
        matrix = np.array([
            [
                np.mean([trial["achievements"].get(name, 0) > 0 for trial in trials])
                for name in achievement_names
            ]
            for trials in results.values()
        ])
        figure, axis = plt.subplots(
            figsize=(max(8, 0.45 * len(achievement_names)), 1 + 0.6 * len(labels))
        )
        image = axis.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        axis.set_xticks(np.arange(len(achievement_names)))
        axis.set_xticklabels(achievement_names, rotation=45, ha="right", fontsize=8)
        axis.set_yticks(np.arange(len(labels)))
        axis.set_yticklabels(labels)
        axis.set_title("Achievement unlock rate (share of trials)")
        figure.colorbar(image, ax=axis, shrink=0.8)
        figure.tight_layout()
        figure.savefig(Path(config.GRAPHS_DIR) / "achievement_heatmap.png", dpi=150)
        plt.close(figure)


# ============================================================
# Entry
# ============================================================
def main():
    results = ResultsStore().all_results()
    if not results:
        return
    make_graphs(results)


if __name__ == "__main__":
    main()
