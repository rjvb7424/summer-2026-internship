import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np


MOVES = ["FORWARD", "BACKWARD", "LEFT", "RIGHT"]
PROVIDER_COLORMAPS = {
    "huggingface": plt.get_cmap("Blues"),
    "openai": plt.get_cmap("Reds"),
    "google": plt.get_cmap("Greens"),
    "anthropic": plt.get_cmap("Purples"),
}
FALLBACK_COLORMAPS = [
    plt.get_cmap("Oranges"),
    plt.get_cmap("Greys"),
    plt.get_cmap("YlGn"),
    plt.get_cmap("PuRd"),
]
SHADE_RANGE = (0.35, 0.85)


def _detect_provider(model_name):
    """Best-effort identification of a model provider from its model ID."""
    name = str(model_name).lower()

    if "/" in name:
        # Most owner/model IDs in these results are from Hugging Face.
        return "huggingface"
    if name.startswith("gpt") or re.match(r"^o\d", name):
        return "openai"
    if name.startswith("gemini"):
        return "google"
    if name.startswith("claude"):
        return "anthropic"
    return "other"


def model_color_map(results):
    """Map each model to a stable colour, grouped by provider family."""
    models = sorted({str(r.get("model_version", "unknown")) for r in results})
    by_provider = defaultdict(list)

    for model in models:
        by_provider[_detect_provider(model)].append(model)

    colours = {}
    fallback_index = 0

    for provider in sorted(by_provider):
        provider_models = sorted(by_provider[provider])
        if provider in PROVIDER_COLORMAPS:
            cmap = PROVIDER_COLORMAPS[provider]
        else:
            cmap = FALLBACK_COLORMAPS[fallback_index % len(FALLBACK_COLORMAPS)]
            fallback_index += 1

        shades = (
            [SHADE_RANGE[1]]
            if len(provider_models) == 1
            else np.linspace(SHADE_RANGE[0], SHADE_RANGE[1], len(provider_models))
        )

        for model, shade in zip(provider_models, shades):
            colours[model] = cmap(shade)

    return colours


def load_results(path):
    """Load and validate the top-level JSON results list."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Run the navigation experiment first."
        )

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Expected the results JSON to contain a top-level list.")

    return [item for item in data if isinstance(item, dict)]


def get_turns(trial):
    """Return the per-turn log, supporting common field names."""
    for key in ("turns", "turn_log", "history", "steps"):
        value = trial.get(key)
        if isinstance(value, list):
            return [turn for turn in value if isinstance(turn, dict)]
    return []


def is_complete_trial(trial):
    """A complete/scorable trial must report whether the goal was reached."""
    return isinstance(trial.get("reached_goal"), bool)


def numeric(value, default=0.0):
    """Safely convert a value to float."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def trial_turn_count(trial):
    """Get turns taken, falling back to the length of the turn log."""
    value = trial.get("turns_taken")
    if value is not None:
        return int(numeric(value))
    return len(get_turns(trial))


def trial_metric(trial, total_field, turn_field):
    """Read a trial-level total, or sum the corresponding turn-level values."""
    value = trial.get(total_field)
    if value is not None:
        return numeric(value)
    return sum(numeric(turn.get(turn_field)) for turn in get_turns(trial))


def trial_validity_counts(trial):
    """Return valid, invalid and unrecognised move counts for one trial."""
    valid = 0
    invalid = 0
    unrecognised = 0

    for turn in get_turns(trial):
        parsed_move = turn.get("parsed_move")
        move_valid = turn.get("move_valid")

        if parsed_move not in MOVES:
            unrecognised += 1

        if move_valid is True:
            valid += 1
        else:
            invalid += 1

    return valid, invalid, unrecognised


def group_by_model(results):
    grouped = defaultdict(list)
    for trial in results:
        grouped[str(trial.get("model_version", "unknown"))].append(trial)
    return dict(grouped)


def save_figure(fig, outdir, filename):
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_success_rate_by_model(results, outdir):
    """Goal-reaching success rate for each model."""
    by_model = group_by_model(results)
    colours = model_color_map(results)

    rows = []
    for model, trials in by_model.items():
        successes = sum(bool(t["reached_goal"]) for t in trials)
        rows.append((model, 100 * successes / len(trials), len(trials)))

    rows.sort(key=lambda row: row[1])
    models = [row[0] for row in rows]
    rates = [row[1] for row in rows]
    counts = [row[2] for row in rows]

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.65 * len(models))))
    bars = ax.barh(models, rates, color=[colours[m] for m in models])

    for bar, rate, count in zip(bars, rates, counts):
        ax.text(
            min(rate + 1.5, 102),
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.1f}% (n={count})",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Trials reaching the goal (%)")
    ax.set_title("Navigation success rate by model")
    ax.set_xlim(0, 112)
    ax.grid(True, axis="x", alpha=0.3)
    save_figure(fig, outdir, "success_rate_by_model.png")


def plot_turns_by_model(results, outdir):
    """Average number of turns, split into successful and failed trials."""
    by_model = group_by_model(results)
    colours = model_color_map(results)
    models = sorted(by_model)

    success_means = []
    failure_means = []

    for model in models:
        successful = [
            trial_turn_count(t) for t in by_model[model] if t["reached_goal"]
        ]
        failed = [
            trial_turn_count(t) for t in by_model[model] if not t["reached_goal"]
        ]
        success_means.append(np.mean(successful) if successful else 0)
        failure_means.append(np.mean(failed) if failed else 0)

    x = np.arange(len(models))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(models)), 4.5))
    ax.bar(
        x - width / 2,
        success_means,
        width,
        label="Successful trials",
        color=[colours[m] for m in models],
    )
    ax.bar(
        x + width / 2,
        failure_means,
        width,
        label="Failed trials",
        color=[colours[m] for m in models],
        alpha=0.45,
        hatch="//",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Average turns")
    ax.set_title("Turns used by outcome and model")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, outdir, "turns_by_model.png")


def plot_move_distribution(results, outdir):
    """Distribution of recognised moves, plus unrecognised responses."""
    by_model = group_by_model(results)
    models = sorted(by_model)
    categories = MOVES + ["UNRECOGNISED"]

    counts = {model: Counter() for model in models}
    for model, trials in by_model.items():
        for trial in trials:
            for turn in get_turns(trial):
                move = turn.get("parsed_move")
                counts[model][move if move in MOVES else "UNRECOGNISED"] += 1

    x = np.arange(len(models))
    width = 0.8 / len(categories)

    fig, ax = plt.subplots(figsize=(max(8, 1.7 * len(models)), 5))
    for index, category in enumerate(categories):
        offsets = x - 0.4 + width / 2 + index * width
        values = [counts[model][category] for model in models]
        ax.bar(offsets, values, width, label=category)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Number of turns")
    ax.set_title("Move distribution by model")
    ax.legend(fontsize=8, ncol=min(5, len(categories)))
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, outdir, "move_distribution_by_model.png")


def plot_move_quality_by_model(results, outdir):
    """Percentage of turns with a valid, invalid or unrecognised move."""
    by_model = group_by_model(results)
    models = sorted(by_model)

    valid_rates = []
    invalid_recognised_rates = []
    unrecognised_rates = []

    for model in models:
        valid = invalid = unrecognised = total = 0

        for trial in by_model[model]:
            turns = get_turns(trial)
            total += len(turns)
            for turn in turns:
                parsed = turn.get("parsed_move")
                if parsed not in MOVES:
                    unrecognised += 1
                elif turn.get("move_valid") is True:
                    valid += 1
                else:
                    invalid += 1

        denominator = total or 1
        valid_rates.append(100 * valid / denominator)
        invalid_recognised_rates.append(100 * invalid / denominator)
        unrecognised_rates.append(100 * unrecognised / denominator)

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(models)), 5))
    ax.bar(models, valid_rates, label="Valid move")
    ax.bar(
        models,
        invalid_recognised_rates,
        bottom=valid_rates,
        label="Recognised but blocked/invalid",
    )
    bottom = np.array(valid_rates) + np.array(invalid_recognised_rates)
    ax.bar(
        models,
        unrecognised_rates,
        bottom=bottom,
        label="Unrecognised or empty response",
    )

    ax.set_ylabel("Share of turns (%)")
    ax.set_title("Move validity by model")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", labelrotation=20)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, outdir, "move_quality_by_model.png")


def plot_success_vs_cost(results, outdir):
    """Success rate against average tokens and elapsed time per trial."""
    by_model = group_by_model(results)
    colours = model_color_map(results)
    models = sorted(by_model)

    fig, (ax_tokens, ax_time) = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, metric_name, xlabel in (
        (ax_tokens, "tokens", "Average total tokens per trial"),
        (ax_time, "time", "Average elapsed seconds per trial"),
    ):
        for model in models:
            trials = by_model[model]
            success_rate = 100 * sum(t["reached_goal"] for t in trials) / len(trials)

            if metric_name == "tokens":
                values = [trial_metric(t, "total_tokens", "total_tokens") for t in trials]
            else:
                values = [
                    trial_metric(t, "total_elapsed_seconds", "elapsed_seconds")
                    for t in trials
                ]

            average = float(np.mean(values)) if values else 0.0
            sample_size = len(trials)

            ax.scatter(
                average,
                success_rate,
                s=max(80, 25 * sample_size),
                color=colours[model],
                alpha=0.8,
                edgecolors="white",
                linewidths=1,
                zorder=3,
            )
            ax.annotate(
                model,
                (average, success_rate),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Success rate (%)")
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)

    ax_tokens.set_title("Success vs tokens")
    ax_time.set_title("Success vs response time")
    fig.suptitle("Efficiency: bubble size = number of trials", fontsize=10)
    save_figure(fig, outdir, "success_vs_cost.png")


def plot_tokens_by_type(results, outdir):
    """Average prompt, output and thinking tokens per trial."""
    by_model = group_by_model(results)
    models = sorted(by_model)

    prompt_values = []
    output_values = []
    thinking_values = []

    for model in models:
        trials = by_model[model]
        prompt_values.append(
            np.mean([trial_metric(t, "total_prompt_tokens", "prompt_tokens") for t in trials])
        )
        output_values.append(
            np.mean([trial_metric(t, "total_output_tokens", "output_tokens") for t in trials])
        )
        thinking_values.append(
            np.mean([trial_metric(t, "total_thinking_tokens", "thinking_tokens") for t in trials])
        )

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(models)), 5))
    ax.bar(models, prompt_values, label="Prompt tokens")
    ax.bar(models, output_values, bottom=prompt_values, label="Output tokens")
    bottom = np.array(prompt_values) + np.array(output_values)
    ax.bar(models, thinking_values, bottom=bottom, label="Thinking tokens")

    ax.set_ylabel("Average tokens per trial")
    ax.set_title("Token usage by model")
    ax.tick_params(axis="x", labelrotation=20)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, outdir, "tokens_by_model.png")


def plot_turn_progression(results, outdir):
    """Average cumulative token and time growth over navigation turns."""
    by_model = group_by_model(results)
    colours = model_color_map(results)

    fig, ax = plt.subplots(figsize=(9, 5))

    for model, trials in sorted(by_model.items()):
        sequences = []
        for trial in trials:
            cumulative = []
            running = 0.0
            for turn in get_turns(trial):
                running += numeric(turn.get("total_tokens"))
                cumulative.append(running)
            if cumulative:
                sequences.append(cumulative)

        if not sequences:
            continue

        max_length = max(map(len, sequences))
        averages = []
        for index in range(max_length):
            available = [seq[index] for seq in sequences if index < len(seq)]
            averages.append(np.mean(available))

        ax.plot(
            range(1, len(averages) + 1),
            averages,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=model,
            color=colours[model],
        )

    ax.set_xlabel("Turn")
    ax.set_ylabel("Average cumulative tokens")
    ax.set_title("Token growth across navigation turns")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save_figure(fig, outdir, "token_growth_by_turn.png")


def build_summary_rows(results):
    """Create one aggregate summary row per model."""
    rows = []

    for model, trials in sorted(group_by_model(results).items()):
        successes = sum(t["reached_goal"] for t in trials)
        turns = [trial_turn_count(t) for t in trials]
        successful_turns = [trial_turn_count(t) for t in trials if t["reached_goal"]]

        all_turns = [turn for trial in trials for turn in get_turns(trial)]
        recognised = sum(turn.get("parsed_move") in MOVES for turn in all_turns)
        valid = sum(turn.get("move_valid") is True for turn in all_turns)

        rows.append(
            {
                "model_version": model,
                "trials": len(trials),
                "successes": successes,
                "success_rate_percent": round(100 * successes / len(trials), 2),
                "avg_turns_all_trials": round(float(np.mean(turns)), 2) if turns else 0,
                "avg_turns_successful_trials": (
                    round(float(np.mean(successful_turns)), 2)
                    if successful_turns
                    else ""
                ),
                "recognised_move_rate_percent": (
                    round(100 * recognised / len(all_turns), 2) if all_turns else 0
                ),
                "valid_move_rate_percent": (
                    round(100 * valid / len(all_turns), 2) if all_turns else 0
                ),
                "avg_elapsed_seconds": round(
                    float(
                        np.mean(
                            [
                                trial_metric(t, "total_elapsed_seconds", "elapsed_seconds")
                                for t in trials
                            ]
                        )
                    ),
                    2,
                ),
                "avg_total_tokens": round(
                    float(
                        np.mean(
                            [
                                trial_metric(t, "total_tokens", "total_tokens")
                                for t in trials
                            ]
                        )
                    ),
                    2,
                ),
                "avg_output_tokens": round(
                    float(
                        np.mean(
                            [
                                trial_metric(t, "total_output_tokens", "output_tokens")
                                for t in trials
                            ]
                        )
                    ),
                    2,
                ),
            }
        )

    return rows


def save_summary_csv(results, outdir):
    rows = build_summary_rows(results)
    if not rows:
        return

    output_path = os.path.join(outdir, "model_summary.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(results):
    rows = build_summary_rows(results)

    print("\nModel summary")
    print("=" * 100)
    for row in rows:
        print(
            f"{row['model_version']}: "
            f"{row['successes']}/{row['trials']} successful "
            f"({row['success_rate_percent']:.1f}%) | "
            f"avg turns {row['avg_turns_all_trials']} | "
            f"valid moves {row['valid_move_rate_percent']:.1f}% | "
            f"avg tokens {row['avg_total_tokens']:.0f} | "
            f"avg time {row['avg_elapsed_seconds']:.1f}s"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Analyse perspective-taking navigation experiment results."
    )
    parser.add_argument(
        "--results",
        default="perspective_taking_results.json",
        help="Path to the navigation results JSON file.",
    )
    parser.add_argument(
        "--outdir",
        default="navigation_plots",
        help="Directory in which graphs and summary CSV are saved.",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    raw_results = load_results(args.results)
    results = [trial for trial in raw_results if is_complete_trial(trial)]

    skipped = len(raw_results) - len(results)
    if skipped:
        print(
            f"Warning: skipped {skipped} incomplete trial(s) without a boolean "
            "'reached_goal' value."
        )

    if not results:
        print("No complete navigation trials found. Nothing to analyse.")
        return

    plot_success_rate_by_model(results, args.outdir)
    plot_turns_by_model(results, args.outdir)
    plot_move_distribution(results, args.outdir)
    plot_move_quality_by_model(results, args.outdir)
    plot_success_vs_cost(results, args.outdir)
    plot_tokens_by_type(results, args.outdir)
    plot_turn_progression(results, args.outdir)
    save_summary_csv(results, args.outdir)
    print_summary(results)

    print(f"\nAnalysis saved to {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    main()
