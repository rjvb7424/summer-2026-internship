import json
import os
from functools import partial
# Internal imports
import cognitive_test
import gemini

# Constants
NUM_TRIALS = 20
RESULTS_FILE = "results.json"
# AI models to use for the cognitive test
# AI models to use for the cognitive test
GEMINI_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
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

def gemini_solver(prompt, model):
    """Call the Gemini API with the given prompt and return the result."""
    return gemini.call_gemini(prompt, model=model)

# load existing results to avoid overwriting previous trials
results = load_existing_results()
start_trial = len(results)

# For each model, figure out how many trials it already has (so re-running
# the script resumes each model independently instead of picking up where
# the last model in the list left off).
for model in GEMINI_MODELS:
    model_results = [r for r in results if r.get("model_version") == model]
    start_trial = len(model_results)
 
    if start_trial >= NUM_TRIALS:
        print(f"\n=== {model}: already has {start_trial}/{NUM_TRIALS} trials, skipping ===")
        continue
 
    print(f"\n### Model: {model} ({start_trial}/{NUM_TRIALS} done) ###")
    solver = partial(gemini_solver, model=model)
 
    for i in range(start_trial, NUM_TRIALS):
        print(f"\n=== {model} | Trial {i+1}/{NUM_TRIALS} ===")
        test = cognitive_test.CognitiveTest()
        result = test.run(num_folds=3, solver=solver)
        result["trial"] = i + 1
        # Tag the result with the model we intended to test, even if the
        # solver call failed and couldn't report its own model_version.
        # This keeps per-model resume counting accurate.
        result["model_version"] = model
        results.append(result)
 
        # save the results after each trial to ensure progress is not lost
        save_results(results)
 
        # print the trial results to the console for immediate feedback
        print(f"Correct: {result['correct_choice']} | {model}: {result['predicted_choice']} | Match: {result['is_correct']}")
 
    model_correct = sum(1 for r in results if r.get("model_version") == model and r["is_correct"])
    model_total = sum(1 for r in results if r.get("model_version") == model)
    print(f"\n{model} accuracy: {model_correct}/{model_total} ({100*model_correct/model_total:.1f}%)")
 
# After all models are complete, print overall accuracy across every trial run so far.
scored = [r for r in results if r.get("is_correct") is not None]
correct = sum(1 for r in scored if r["is_correct"])
print(f"\n=== Overall accuracy across all models: {correct}/{len(scored)} ({100*correct/len(scored):.1f}%) ===")
 
# For each model, figure out how many trials it already has (so re-running
# the script resumes each model independently instead of picking up where
# the last model in the list left off).
for model in GEMINI_MODELS:
    model_results = [r for r in results if r.get("model_version") == model]
    start_trial = len(model_results)
 
    if start_trial >= NUM_TRIALS:
        print(f"\n=== {model}: already has {start_trial}/{NUM_TRIALS} trials, skipping ===")
        continue
 
    print(f"\n### Model: {model} ({start_trial}/{NUM_TRIALS} done) ###")
    solver = partial(gemini_solver, model=model)
 
    for i in range(start_trial, NUM_TRIALS):
        print(f"\n=== {model} | Trial {i+1}/{NUM_TRIALS} ===")
        test = cognitive_test.CognitiveTest()
        result = test.run(num_folds=3, solver=solver)
        result["trial"] = i + 1
        # Tag the result with the model we intended to test, even if the
        # solver call failed and couldn't report its own model_version.
        # This keeps per-model resume counting accurate.
        result["model_version"] = model
        results.append(result)
 
        # save the results after each trial to ensure progress is not lost
        save_results(results)
 
        # print the trial results to the console for immediate feedback
        print(f"Correct: {result['correct_choice']} | {model}: {result['predicted_choice']} | Match: {result['is_correct']}")
 
    model_correct = sum(1 for r in results if r.get("model_version") == model and r["is_correct"])
    model_total = sum(1 for r in results if r.get("model_version") == model)
    print(f"\n{model} accuracy: {model_correct}/{model_total} ({100*model_correct/model_total:.1f}%)")
 
# After all models are complete, print overall accuracy across every trial run so far.
scored = [r for r in results if r.get("is_correct") is not None]
correct = sum(1 for r in scored if r["is_correct"])
print(f"\n=== Overall accuracy across all models: {correct}/{len(scored)} ({100*correct/len(scored):.1f}%) ===")
 