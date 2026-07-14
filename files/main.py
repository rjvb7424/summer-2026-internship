import json
import os
import re
from functools import partial
# Internal imports
import analyze_results
import huggingface
import gemini
import gpt
from files.crafter.crafter_test import CrafterTest

from config import (
    NUM_TRIALS,
    MAX_STEPS,
    BASE_SEED,
    RESULTS_FILE,
    RECORD_DIRECTORY,
    SHOW_SIMULATION,
    RECORD_VIDEO,
    SAVE_PROMPTS,
    RUN_ANALYSIS,
    RUN_GEMINI,
    RUN_GPT,
    RUN_HUGGINGFACE,
    GEMINI_MODELS,
    GPT_MODELS,
    HUGGINGFACE_MODELS,
)

def load_existing_results():
    """Load existing results if present, otherwise return an empty list."""
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r", encoding="utf-8") as file:
                return json.load(file)
        except json.JSONDecodeError:
            print(f"Warning: {RESULTS_FILE} is empty or corrupted. Starting fresh.")
    return []

def save_results(results):
    """Save all experiment results after each episode."""
    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

# Some models may have characters that are not safe for directory names.
# Therefore we sanitize the model identifier to create a safe directory name.
# This is used in the "recordings" directory to store video recordings for each model and trial.
def safe_directory_name(value):
    """Convert a model identifier into a safe directory name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)

def build_models_to_run():
    """Build the enabled list of model and solver pairs."""
    models_to_run = []
    if RUN_GEMINI:
        models_to_run += [(model, gemini.call_gemini) for model in GEMINI_MODELS]
    if RUN_GPT:
        models_to_run += [(model, gpt.call_gpt) for model in GPT_MODELS]
    if RUN_HUGGINGFACE:
        models_to_run += [(model, huggingface.call_huggingface) for model in HUGGINGFACE_MODELS]
    return models_to_run

def main():
    # Load existing results and determine which models to run based on the configuration.
    models_to_run = build_models_to_run()
    if not models_to_run:
        print("No providers enabled!")
        return
    results = load_existing_results()

    for model, solver_fn in models_to_run:
        # For each model, check how many trials have already been completed and skip those.
        model_results = [ result for result in results if result.get("model_version") == model]
        start_trial = len(model_results)
        if start_trial >= NUM_TRIALS:
            print(f"\n=== {model}: already has {start_trial}/{NUM_TRIALS}, skipping ===")
            continue
        print(f"\n=== Model: {model} ({start_trial}/{NUM_TRIALS} done) ===")
        solver = partial(solver_fn, model=model)

        # For each trial, run the Crafter test and record the results.
        for trial_index in range(start_trial, NUM_TRIALS):
            trial_number = trial_index + 1
            seed = BASE_SEED + trial_index
            print(f"\n=== {model} | Trial {trial_number}/{NUM_TRIALS} ===")

            record_directory = None
            if RECORD_VIDEO:
                record_directory = os.path.join(
                    RECORD_DIRECTORY,
                    safe_directory_name(model),
                    f"trial_{trial_number:03d}",
                )

            test = CrafterTest(
                max_steps=MAX_STEPS,
                seed=seed,
                show_simulation=SHOW_SIMULATION,
                record_directory=record_directory,
                save_prompts=SAVE_PROMPTS,
            )

            try:
                result = test.run(
                    solver=solver,
                    model=model,
                    trial=trial_number,
                )
            except Exception as error:
                result = {
                    "model_version": model,
                    "trial": trial_number,
                    "seed": seed,
                    "error": str(error),
                    "solver_failed": True,
                }
                print(f"[{model}] Trial failed: {error}")

            results.append(result)
            save_results(results)

            print(
                f"Reward: {result.get('total_reward', 0):.1f} | "
                f"Achievements: {result.get('num_achievements', 0)} | "
                f"Steps: {result.get('episode_steps', 0)}"
            )

            if result.get("stopped_by_user"):
                print("Simulation window closed. Stopping the experiment.")
                if RUN_ANALYSIS:
                    analyze_results.run()
                return

        completed = [
            result
            for result in results
            if result.get("model_version") == model
            and result.get("error") is None
        ]
        # If there are completed trials, calculate and print the average reward and achievements for the model.
        if completed:
            average_reward = sum(
                result.get("total_reward", 0) for result in completed
            ) / len(completed)
            average_achievements = sum(
                result.get("num_achievements", 0) for result in completed
            ) / len(completed)
            print(
                f"\n{model} averages: reward={average_reward:.2f}, "
                f"achievements={average_achievements:.2f}"
            )
    # If all trials for all models are completed, run the analysis if enabled.
    if RUN_ANALYSIS:
        analyze_results.main()

if __name__ == "__main__":
    main()
