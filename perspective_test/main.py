import json
import os
from functools import partial

# Internal imports
from navigation_experiment import NavigationTrial
import huggingface

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

NUM_TRIALS = 30
RESULTS_FILE = "perspective_taking_results.json"

# Size of the generated archipelago (size x size tiles).
MAP_SIZE = 10

# How many turns the AI gets before a trial counts as a failure. None means
# "let NavigationTrial pick its own default" - a multiple of the straight-
# line distance between start and goal (see DEFAULT_TURNS_PER_TILE and
# MINIMUM_MAX_TURNS in navigation_experiment.py). Set an integer here (e.g.
# MAX_TURNS = 50) to use a fixed budget for every trial instead.
MAX_TURNS = None

# How many new tokens each model call is allowed to generate. Phi-2 tends
# to ramble past its actual answer (it's a base/completion model, not
# instruction-tuned), so this is kept modest rather than the thousands of
# tokens a reasoning model like DeepSeek-R1 would need.
HUGGINGFACE_MAX_NEW_TOKENS = 256

# Prints each model's tokens to the terminal live as they're generated.
# Turn this off if you want quieter logs once you trust it's working.
HUGGINGFACE_STREAM_OUTPUT = True

# Any model ID from huggingface.co goes here - the script runs every model
# in this list through the same navigation experiment, one after another.
HUGGINGFACE_MODELS = [
    "microsoft/phi-2",
]


def load_existing_results():
    """Load existing results from the JSON file if it exists, otherwise return an empty list."""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    return []


def save_results(results):
    """Save the results to a JSON file."""
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def huggingface_solver(prompt, model):
    """Call a local Hugging Face model with the given prompt and return the result."""
    return huggingface.call_huggingface(
        prompt, model=model, show_live_output=HUGGINGFACE_STREAM_OUTPUT
    )


# Every model in HUGGINGFACE_MODELS uses the same solver function - there's
# only one provider now, so (unlike the old per-provider RUN_X toggles) the
# list itself is the only thing you need to edit to add or remove a model.
models_to_run = [(model, huggingface_solver) for model in HUGGINGFACE_MODELS]

if not models_to_run:
    print("No models configured! Add at least one to HUGGINGFACE_MODELS.")
    exit()

# load existing results to avoid overwriting previous trials
results = load_existing_results()

# For each model, figure out how many trials it already has (so re-running
# the script resumes each model independently instead of picking up where
# the last model in the list left off).
for model, solver_fn in models_to_run:
    model_results = [r for r in results if r.get("model_version") == model]
    start_trial = len(model_results)
    if start_trial >= NUM_TRIALS:
        print(f"\n=== {model}: already has {start_trial}/{NUM_TRIALS} trials, skipping ===")
        continue

    print(f"\n### Model: {model} ({start_trial}/{NUM_TRIALS} done) ###")
    solver = partial(solver_fn, model=model)

    for i in range(start_trial, NUM_TRIALS):
        print(f"\n=== {model} | Trial {i + 1}/{NUM_TRIALS} ===")
        trial = NavigationTrial(map_size=MAP_SIZE, max_turns=MAX_TURNS)
        result = trial.run(solver=solver)
        result["trial"] = i + 1
        # Tag the result with the model we intended to test, even if the
        # solver call failed partway through and couldn't finish reporting
        # its own model_version. This keeps per-model resume counting accurate.
        result["model_version"] = model
        results.append(result)

        # save the results after each trial to ensure progress is not lost
        save_results(results)

        # print the trial results to the console for immediate feedback
        outcome = "REACHED GOAL" if result["reached_goal"] else "FAILED (timed out)"
        print(f"{outcome} in {result['turns_taken']}/{result['max_turns']} turns | {model}")

    model_successes = sum(1 for r in results if r.get("model_version") == model and r["reached_goal"])
    model_total = sum(1 for r in results if r.get("model_version") == model)
    print(f"\n{model} success rate: {model_successes}/{model_total} "
          f"({100 * model_successes / model_total:.1f}%)")

# After all models are complete, print overall success rate across every trial run so far.
scored = [r for r in results if "reached_goal" in r]
if scored:
    successes = sum(1 for r in scored if r["reached_goal"])
    print(f"\n=== Overall success rate across all models: {successes}/{len(scored)} "
          f"({100 * successes / len(scored):.1f}%) ===")