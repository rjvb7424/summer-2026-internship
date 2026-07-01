"""
analyze_results.py

Reads results.json (produced by run.py) and regenerates a set of graphs
in the plots/ folder. Safe to re-run any time you add more trials --
it always reflects whatever is currently in results.json.

Usage:
    python analyze_results.py
    python analyze_results.py --results results.json --outdir plots
"""

import argparse
import json
import os
from collections import Counter

import matplotlib.pyplot as plt

LETTERS = ["A", "B", "C", "D", "E"]

# Fixed color order so each model keeps the same color across runs/graphs,
# regardless of the order models happen to appear in results.json.
MODEL_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#e34948", "#4a3aa7", "#e87ba4", "#eb6834"]


def model_color_map(results):
    models = sorted({r.get("model_version", "unknown") for r in results})
    return {m: MODEL_COLORS[i % len(MODEL_COLORS)] for i, m in enumerate(models)}


def load_results(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}. Run run.py first.")
    with open(path, "r") as f:
        return json.load(f)


def filter_scored(results):
    """Only keep trials that actually got a solver response."""
    return [r for r in results if r.get("predicted_choice") is not None]


def plot_rolling_accuracy(results, outdir):
    """Accuracy so far, trial by trial. Shows whether performance is
    trending up, down, or flat as you collect more data."""
    correct_flags = [1 if r["is_correct"] else 0 for r in results]
    running_acc = []
    total_correct = 0
    for i, c in enumerate(correct_flags, start=1):
        total_correct += c
        running_acc.append(100 * total_correct / i)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(running_acc) + 1), running_acc, marker="o", markersize=3, linewidth=1.5, color="#2a78d6")
    ax.axhline(20, linestyle="--", linewidth=1, color="#898781", label="Chance (20%, 5 choices)")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Cumulative accuracy (%)")
    ax.set_title("Rolling accuracy over trials")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "rolling_accuracy.png"), dpi=150)
    plt.close(fig)


def plot_letter_distribution(results, outdir):
    """Predicted letter vs correct letter counts. A skew toward one
    letter in 'predicted' that doesn't match 'correct' is a sign of
    positional bias rather than real reasoning."""
    predicted_counts = Counter(r["predicted_choice"] for r in results)
    correct_counts = Counter(r["correct_choice"] for r in results)

    x = range(len(LETTERS))
    width = 0.35
    predicted_vals = [predicted_counts.get(l, 0) for l in LETTERS]
    correct_vals = [correct_counts.get(l, 0) for l in LETTERS]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - width / 2 for i in x], predicted_vals, width, label="Predicted", color="#2a78d6")
    ax.bar([i + width / 2 for i in x], correct_vals, width, label="Correct", color="#eda100")
    ax.set_xticks(list(x))
    ax.set_xticklabels(LETTERS)
    ax.set_ylabel("Count")
    ax.set_title("Predicted vs correct answer letter")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "letter_distribution.png"), dpi=150)
    plt.close(fig)


def plot_tokens_per_trial(results, outdir):
    """Output tokens per trial. Consistently low token counts (e.g. 1)
    suggest the model is answering without any visible reasoning."""
    trial_ids = [r.get("trial", i + 1) for i, r in enumerate(results)]
    tokens = [r.get("output_tokens") or 0 for r in results]
    colors = ["#1baf7a" if r["is_correct"] else "#2a78d6" for r in results]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(trial_ids, tokens, color=colors)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Output tokens")
    ax.set_yscale("log")
    ax.set_title("Output tokens per trial (green = correct)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "tokens_per_trial.png"), dpi=150)
    plt.close(fig)


def plot_elapsed_time(results, outdir):
    """Response latency per trial. Useful for spotting when the model
    actually 'thinks' (slower) vs instant-guesses."""
    trial_ids = [r.get("trial", i + 1) for i, r in enumerate(results)]
    elapsed = [r.get("elapsed_seconds") or 0 for r in results]
    colors = ["#1baf7a" if r["is_correct"] else "#2a78d6" for r in results]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(trial_ids, elapsed, color=colors)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Elapsed seconds")
    ax.set_title("Response time per trial (green = correct)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "elapsed_time.png"), dpi=150)
    plt.close(fig)


def plot_accuracy_by_num_folds(results, outdir):
    """Accuracy broken down by number of folds, in case difficulty
    scales with fold count."""
    by_folds = {}
    for r in results:
        n = r.get("num_folds")
        by_folds.setdefault(n, []).append(1 if r["is_correct"] else 0)

    fold_counts = sorted(by_folds.keys())
    accuracies = [100 * sum(by_folds[n]) / len(by_folds[n]) for n in fold_counts]
    sample_sizes = [len(by_folds[n]) for n in fold_counts]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([str(n) for n in fold_counts], accuracies, color="#4a3aa7")
    for bar, n_samples in zip(bars, sample_sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"n={n_samples}", ha="center", fontsize=9, color="#52514e")
    ax.axhline(20, linestyle="--", linewidth=1, color="#898781", label="Chance (20%)")
    ax.set_xlabel("Number of folds")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by fold count")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "accuracy_by_num_folds.png"), dpi=150)
    plt.close(fig)


def plot_accuracy_by_model(results, outdir):
    """Accuracy broken down by model, the main comparison once you're
    testing more than one model on the same benchmark."""
    by_model = {}
    for r in results:
        m = r.get("model_version", "unknown")
        by_model.setdefault(m, []).append(1 if r["is_correct"] else 0)

    models = sorted(by_model.keys())
    accuracies = [100 * sum(by_model[m]) / len(by_model[m]) for m in models]
    sample_sizes = [len(by_model[m]) for m in models]
    colors = [model_color_map(results)[m] for m in models]

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(models)), 4))
    bars = ax.bar(models, accuracies, color=colors)
    for bar, n_samples in zip(bars, sample_sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"n={n_samples}", ha="center", fontsize=9, color="#52514e")
    ax.axhline(20, linestyle="--", linewidth=1, color="#898781", label="Chance (20%)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by model")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", labelrotation=15)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "accuracy_by_model.png"), dpi=150)
    plt.close(fig)


def plot_rolling_accuracy_by_model(results, outdir):
    """Same rolling-accuracy view as plot_rolling_accuracy, but one line
    per model so you can see how they trend as trials accumulate."""
    by_model = {}
    for r in results:
        m = r.get("model_version", "unknown")
        by_model.setdefault(m, []).append(1 if r["is_correct"] else 0)

    colors = model_color_map(results)
    fig, ax = plt.subplots(figsize=(8, 4))
    for m in sorted(by_model.keys()):
        flags = by_model[m]
        running_acc = []
        total_correct = 0
        for i, c in enumerate(flags, start=1):
            total_correct += c
            running_acc.append(100 * total_correct / i)
        ax.plot(range(1, len(running_acc) + 1), running_acc, marker="o", markersize=3,
                linewidth=1.5, color=colors[m], label=m)

    ax.axhline(20, linestyle="--", linewidth=1, color="#898781", label="Chance (20%)")
    ax.set_xlabel("Trial (within model)")
    ax.set_ylabel("Cumulative accuracy (%)")
    ax.set_title("Rolling accuracy by model")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "rolling_accuracy_by_model.png"), dpi=150)
    plt.close(fig)


def plot_elapsed_time_by_model(results, outdir):
    """Average response time per model, since slower/thinking models
    and fast/lite models are worth comparing directly."""
    by_model = {}
    for r in results:
        m = r.get("model_version", "unknown")
        by_model.setdefault(m, []).append(r.get("elapsed_seconds") or 0)

    models = sorted(by_model.keys())
    avg_elapsed = [sum(by_model[m]) / len(by_model[m]) for m in models]
    colors = [model_color_map(results)[m] for m in models]

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(models)), 4))
    ax.bar(models, avg_elapsed, color=colors)
    ax.set_xlabel("Model")
    ax.set_ylabel("Avg elapsed seconds")
    ax.set_title("Average response time by model")
    ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "elapsed_time_by_model.png"), dpi=150)
    plt.close(fig)


def print_summary(results):
    n = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    reasoning_trials = sum(1 for r in results if (r.get("output_tokens") or 0) > 5)
    print(f"Trials scored: {n}")
    print(f"Accuracy: {correct}/{n} ({100 * correct / n:.1f}%)")
    print(f"Trials with >5 output tokens (visible reasoning): {reasoning_trials}/{n}")
    pred_counts = Counter(r["predicted_choice"] for r in results)
    print("Predicted letter distribution:", dict(pred_counts))

    by_model = {}
    for r in results:
        m = r.get("model_version", "unknown")
        by_model.setdefault(m, []).append(r)
    if len(by_model) > 1:
        print("\nBy model:")
        for m in sorted(by_model.keys()):
            rs = by_model[m]
            c = sum(1 for r in rs if r["is_correct"])
            print(f"  {m}: {c}/{len(rs)} ({100 * c / len(rs):.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results.json", help="Path to results.json")
    parser.add_argument("--outdir", default="plots", help="Directory to save graphs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    raw_results = load_results(args.results)
    results = filter_scored(raw_results)

    if not results:
        print("No scored trials found yet (predicted_choice is None for all). Nothing to plot.")
        return

    plot_rolling_accuracy(results, args.outdir)
    plot_letter_distribution(results, args.outdir)
    plot_tokens_per_trial(results, args.outdir)
    plot_elapsed_time(results, args.outdir)
    plot_accuracy_by_num_folds(results, args.outdir)
    plot_accuracy_by_model(results, args.outdir)
    plot_rolling_accuracy_by_model(results, args.outdir)
    plot_elapsed_time_by_model(results, args.outdir)

    print_summary(results)
    print(f"\nGraphs saved to {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    main()