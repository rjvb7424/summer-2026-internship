import argparse
import json
import os
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# The letters used for the candidates in the spatial visualisation test.
LETTERS = ["A", "B", "C", "D", "E"]

# Each AI provider gets their own color family (a matplotlib colormap). 
# Models from the same provider are shaded from light to dark within that family.
PROVIDER_COLORMAPS = {
    "gemini": plt.get_cmap("Blues"),
    "gpt": plt.get_cmap("Reds"),
}
# Fallback colormaps for providers not in PROVIDER_COLORMAPS.
FALLBACK_COLORMAPS = [plt.get_cmap("Greens"), plt.get_cmap("Purples"), plt.get_cmap("Oranges"), plt.get_cmap("Greys")]
# The range of shades to use for models within a provider's colormap
SHADE_RANGE = (0.35, 0.85)

def _detect_provider(model_name):
    """Best-effort guess at which company a model belongs to, from its name."""
    name = model_name.lower()
    if name.startswith("gemini"):
        return "gemini"
    if name.startswith("gpt") or re.match(r"^o\d", name):
        return "gpt"
    return "other"

def model_color_map(results):
    """Maps each model name to a color, grouped by provider color family"""
    models = sorted({r.get("model_version", "unknown") for r in results})
    
    # Group models by provider, so we can shade them within their provider's colormap.
    by_provider = {}
    for m in models:
        by_provider.setdefault(_detect_provider(m), []).append(m)

    color_map = {}
    fallback_idx = 0
    # Sort provider keys so fallback-family assignment is stable across runs.
    for provider in sorted(by_provider.keys()):
        provider_models = sorted(by_provider[provider])
        n = len(provider_models)

        if provider in PROVIDER_COLORMAPS:
            cmap = PROVIDER_COLORMAPS[provider]
        else:
            cmap = FALLBACK_COLORMAPS[fallback_idx % len(FALLBACK_COLORMAPS)]
            fallback_idx += 1

        shades = [SHADE_RANGE[1]] if n == 1 else np.linspace(SHADE_RANGE[0], SHADE_RANGE[1], n)
        for model, shade in zip(provider_models, shades):
            color_map[model] = cmap(shade)

    return color_map

def load_results(path):
    """Load results from a JSON file, raising an error if the file doesn't exist."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}. Run run.py first.")
    with open(path, "r") as f:
        return json.load(f)

def filter_scored(results):
    """Only keep trials that actually got a solver response."""
    return [r for r in results if r.get("predicted_choice") is not None]

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

def plot_accuracy_by_model(results, outdir):
    """Accuracy broken down by model, ranked best-to-worst.""" 
    """Useful for comparing model performance."""
    by_model = {}
    for r in results:
        m = r.get("model_version", "unknown")
        by_model.setdefault(m, []).append(1 if r["is_correct"] else 0)

    # Sort the models by accuracy, so that the best model ends up at the top of the chart.
    # the best model ends up at the top of the chart.
    ranked = sorted(by_model.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
    models = [m for m, _ in ranked]
    accuracies = [100 * sum(flags) / len(flags) for _, flags in ranked]
    sample_sizes = [len(flags) for _, flags in ranked]
    colors = [model_color_map(results)[m] for m in models]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(models))))
    bars = ax.barh(models, accuracies, color=colors)
    for bar, acc, n_samples in zip(bars, accuracies, sample_sizes):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{acc:.0f}%", va="center", fontsize=9, color="#52514e")
    ax.axvline(20, linestyle="--", linewidth=1, color="#898781", label="Chance (20%)")
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Model against accuracy")
    ax.set_xlim(0, 110)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "accuracy_by_model.png"), dpi=150)
    plt.close(fig)

def plot_accuracy_vs_cost(results, outdir):
    """The efficiency view: accuracy vs how much the model spent to get
    there (tokens and time). One point per model, bubble size = number
    of trials. Top-left = accurate AND cheap (good). Bottom-right =
    expensive AND wrong (bad)."""
    by_model = {}
    for r in results:
        m = r.get("model_version", "unknown")
        by_model.setdefault(m, []).append(r)

    models = sorted(by_model.keys())
    colors = model_color_map(results)

    fig, (ax_tokens, ax_time) = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, field, xlabel in [
        (ax_tokens, "total_tokens", "Avg total tokens per trial"),
        (ax_time, "elapsed_seconds", "Avg response time (s) per trial"),
    ]:
        for m in models:
            rs = by_model[m]
            acc = 100 * sum(1 for r in rs if r["is_correct"]) / len(rs)
            avg_cost = sum((r.get(field) or 0) for r in rs) / len(rs)
            n = len(rs)
            ax.scatter(avg_cost, acc, s=max(80, 25 * n), color=colors[m],
                       alpha=0.8, edgecolors="white", linewidths=1, zorder=3)
            ax.annotate(m, (avg_cost, acc), textcoords="offset points",
                        xytext=(6, 6), fontsize=8, color="#52514e")
        ax.axhline(20, linestyle="--", linewidth=1, color="#898781")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)

    ax_tokens.set_title("Accuracy vs tokens spent")
    ax_time.set_title("Accuracy vs time spent")
    fig.suptitle("Efficiency: bubble size = number of trials", fontsize=10, color="#898781")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "accuracy_vs_cost.png"), dpi=150)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results.json", help="Path to results.json")
    parser.add_argument("--outdir", default="plots", help="Directory to save graphs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load the results and filter to only scored trials (those with a predicted choice).
    raw_results = load_results(args.results)
    results = filter_scored(raw_results)

    if not results:
        print("No scored trials found yet. Nothing to plot!")
        return

    plot_letter_distribution(results, args.outdir)
    plot_accuracy_by_model(results, args.outdir)
    plot_accuracy_vs_cost(results, args.outdir)
    plot_elapsed_time_by_model(results, args.outdir)

    print(f"\nGraphs saved to {os.path.abspath(args.outdir)}/")

if __name__ == "__main__":
    main()
