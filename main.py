from functools import partial

import analyze_results
import gemini
import gpt
import huggingface
from crafter_test import CrafterTest
from results_store import ResultsStore

from config import (
    NUM_TRIALS,
    MAX_STEPS,
    BASE_SEED,
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
    models_to_run = build_models_to_run()
    if not models_to_run:
        print("No providers enabled!")
        return
    store = ResultsStore()

    for model, solver_fn in models_to_run:
        # Resume: skip trials that already have a stored result.
        start_trial = store.completed_trials(model)
        if start_trial >= NUM_TRIALS:
            print(f"\n=== {model}: already has {start_trial}/{NUM_TRIALS}, skipping ===")
            continue
        print(f"\n=== Model: {model} ({start_trial}/{NUM_TRIALS} done) ===")
        solver = partial(solver_fn, model=model)

        for trial_index in range(start_trial, NUM_TRIALS):
            trial_number = trial_index + 1
            seed = BASE_SEED + trial_index
            print(f"\n=== {model} | Trial {trial_number}/{NUM_TRIALS} ===")

            test = CrafterTest(
                max_steps=MAX_STEPS,
                seed=seed,
                show_simulation=SHOW_SIMULATION,
                record_video=RECORD_VIDEO,
                save_prompts=SAVE_PROMPTS,
            )
            result = test.run(solver=solver, model=model, trial=trial_number)
            store.append_trial(model, result)

            print(
                f"Reward: {result['total_reward']:.1f} | "
                f"Achievements: {result['achievements_unlocked']} | "
                f"Steps: {result['steps']}"
            )

            if result["stopped_by_user"]:
                print("Simulation window closed. Stopping the experiment.")
                if RUN_ANALYSIS:
                    analyze_results.main()
                return

        trials = store.load_trials(model)
        average_reward = sum(t["total_reward"] for t in trials) / len(trials)
        average_achievements = sum(t["achievements_unlocked"] for t in trials) / len(trials)
        print(
            f"\n{model} averages: reward={average_reward:.2f}, "
            f"achievements={average_achievements:.2f}"
        )

        # Free MPS memory before the next local model.
        if solver_fn is huggingface.call_huggingface:
            huggingface.unload_models()

    if RUN_ANALYSIS:
        analyze_results.main()

if __name__ == "__main__":
    main()