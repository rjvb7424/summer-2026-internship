import json
import os
import re
from functools import partial
# Internal imports
import analyze_results
import gemini
import gpt
import huggingface
from crafter_test import CrafterTest

# Experiment configuration constants.
NUM_TRIALS = 5
MAX_STEPS = 200
BASE_SEED = 20260708
RESULTS_FILE = "results.json"
RECORD_DIRECTORY = "recordings"

# Toggle experiment features.
SHOW_SIMULATION = True
RECORD_VIDEO = True
SAVE_PROMPTS = False
RUN_ANALYSIS = True

# Toggle which AI providers to run in the experiment.
RUN_GEMINI = False
RUN_GPT = False
RUN_HUGGINGFACE = True

# AI models to use for the Crafter experiment.
GEMINI_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
]
GPT_MODELS = [
    "o3-2025-04-16",
    "gpt-5-nano-2025-08-07",
    "gpt-5.5-2026-04-23",
    "gpt-5.4-mini-2026-03-17",
]
HUGGINGFACE_MODELS = [
    "microsoft/Phi-4-mini-instruct",
]

def load_existing_results():
    """Load existing results if present, otherwise return an empty list."""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    return []

def save_results(results):
    """Save all experiment results after each episode."""
    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

def flatten_messages(messages):
    """Convert chat messages to a plain prompt for non-chat wrappers."""
    if isinstance(messages, str):
        return messages

    return "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}"
        for message in messages
    )

def gemini_solver(messages, model, candidate_actions=None):
    """Call Gemini using the same system/state prompt content."""
    return gemini.call_gemini(
        flatten_messages(messages),
        model=model,
    )

def gpt_solver(messages, model, candidate_actions=None):
    """Call GPT using the same system/state prompt content."""
    return gpt.call_gpt(
        flatten_messages(messages),
        model=model,
    )

def huggingface_solver(messages, model, candidate_actions=None):
    """Let the local model score the state-aware action candidates."""
    return huggingface.call_huggingface(
        messages=messages,
        model=model,
        candidate_actions=candidate_actions,
    )

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
        models_to_run += [(model, gemini_solver) for model in GEMINI_MODELS]
    if RUN_GPT:
        models_to_run += [(model, gpt_solver) for model in GPT_MODELS]
    if RUN_HUGGINGFACE:
        models_to_run += [
            (model, huggingface_solver) for model in HUGGINGFACE_MODELS
        ]
    return models_to_run

def main():
    models_to_run = build_models_to_run()
    # Check if any providers are enabled; if not, print a message and exit.
    if not models_to_run:
        print("No providers enabled!")
        return

    results = load_existing_results()

    # For each model, run the specified number of trials, skipping any that have already been completed.
    for model, solver_fn in models_to_run:
        model_results = [
            result
            for result in results
            if result.get("model_version") == model
        ]
        start_trial = len(model_results)

        if start_trial >= NUM_TRIALS:
            print(
                f"\n=== {model}: already has {start_trial}/{NUM_TRIALS} "
                "trials, skipping ==="
            )
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
        analyze_results.run()

if __name__ == "__main__":
    main()
