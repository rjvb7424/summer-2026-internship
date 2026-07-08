import argparse
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


def load_results(path):
    """Load Crafter results from a JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}. Run run_crafter.py first.")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def filter_completed(results):
    """Keep episodes that contain usable Crafter metrics."""
    return [
        result
        for result in results
        if result.get("error") is None
        and "total_reward" in result
        and "achievements" in result
    ]


def group_by_model(results):
    """Group result dictionaries by model identifier."""
    grouped = {}
    for result in results:
        model = result.get("model_version", "unknown")
        grouped.setdefault(model, []).append(result)
    return grouped


def calculate_achievement_success(results):
    """Calculate the percentage of episodes unlocking each achievement."""
    achievement_names = sorted({
        name
        for result in results
        for name in result.get("achievements", {})
    })
    success = {}

    for achievement in achievement_names:
        unlocked = sum(
            1
            for result in results
            if result.get("achievements", {}).get(achievement, 0) > 0
        )
        success[achievement] = 100 * unlocked / len(results)
    return success


def calculate_crafter_score(success_rates):
    """Calculate the geometric mean score used for Crafter achievements."""
    if not success_rates:
        return 0.0
    values = np.array(list(success_rates.values()), dtype=float)
    return float(np.exp(np.mean(np.log1p(values))) - 1)


def plot_model_summary(results, outdir):
    """Plot average reward, achievements, and Crafter score by model."""
    grouped = group_by_model(results)
    models = sorted(grouped)

    average_rewards = []
    average_achievements = []
    crafter_scores = []

    for model in models:
        model_results = grouped[model]
        average_rewards.append(np.mean([
            result["total_reward"] for result in model_results
        ]))
        average_achievements.append(np.mean([
            result["num_achievements"] for result in model_results
        ]))
        success = calculate_achievement_success(model_results)
        crafter_scores.append(calculate_crafter_score(success))

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(models)), 5))
    ax.bar(x - width, average_rewards, width, label="Avg reward")
    ax.bar(x, average_achievements, width, label="Avg achievements")
    ax.bar(x + width, crafter_scores, width, label="Crafter score")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_title("Crafter performance by model")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "crafter_model_summary.png"), dpi=150)
    plt.close(fig)


def plot_achievement_heatmap(results, outdir):
    """Plot per-model achievement success rates."""
    grouped = group_by_model(results)
    models = sorted(grouped)
    achievements = sorted({
        name
        for result in results
        for name in result.get("achievements", {})
    })

    matrix = []
    for model in models:
        success = calculate_achievement_success(grouped[model])
        matrix.append([success.get(name, 0) for name in achievements])

    fig, ax = plt.subplots(
        figsize=(max(10, 0.55 * len(achievements)), max(3, 0.7 * len(models)))
    )
    image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(achievements)))
    ax.set_xticklabels(achievements, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_title("Achievement success rate (%)")
    fig.colorbar(image, ax=ax, label="Success rate (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "crafter_achievement_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_action_distribution(results, outdir):
    """Plot how often each model selected each Crafter action."""
    grouped = group_by_model(results)
    models = sorted(grouped)
    actions = sorted({
        decision.get("action")
        for result in results
        for decision in result.get("trajectory", [])
        if decision.get("action")
    })

    x = np.arange(len(actions))
    width = 0.8 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(max(10, 0.75 * len(actions)), 5))
    for index, model in enumerate(models):
        counts = Counter(
            decision.get("action")
            for result in grouped[model]
            for decision in result.get("trajectory", [])
        )
        values = [counts.get(action, 0) for action in actions]
        offset = (index - (len(models) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(actions, rotation=45, ha="right")
    ax.set_ylabel("Action count")
    ax.set_title("Action distribution by model")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "crafter_action_distribution.png"), dpi=150)
    plt.close(fig)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="crafter_results.json",
        help="Path to crafter_results.json",
    )
    parser.add_argument(
        "--outdir",
        default="crafter_plots",
        help="Directory to save graphs",
    )
    args, _ = parser.parse_known_args()

    os.makedirs(args.outdir, exist_ok=True)
    results = filter_completed(load_results(args.results))

    if not results:
        print("No completed Crafter trials found. Nothing to plot!")
        return

    plot_model_summary(results, args.outdir)
    plot_achievement_heatmap(results, args.outdir)
    plot_action_distribution(results, args.outdir)
    print(f"\nCrafter graphs saved to {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    run()
