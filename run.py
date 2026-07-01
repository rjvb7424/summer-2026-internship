import json
import os
# Internal imports
import cognitive_test
import gemini

# Constants
NUM_TRIALS = 3
RESULTS_FILE = "results.json"

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

def gemini_solver(prompt):
    """Call the Gemini API with the given prompt and return the result."""
    return gemini.call_gemini(prompt)

# load existing results to avoid overwriting previous trials
results = load_existing_results()
start_trial = len(results)

# For each trial, create a new CognitiveTest instance, run it, and save the results.
for i in range(start_trial, NUM_TRIALS):
    print(f"\n=== Trial {i+1}/{NUM_TRIALS} ===")
    test = cognitive_test.CognitiveTest()
    result = test.run(num_folds=3, solver=gemini_solver)
    result["trial"] = i + 1
    results.append(result)
    # save the results after each trial to ensure progress is not lost
    save_results(results)
    # print the trial results to the console for immediate feedback
    print(f"Correct: {result['correct_choice']} | Gemini: {result['predicted_choice']} | Match: {result['is_correct']}")
# After all trials are complete, calculate and print the overall accuracy of the Gemini model on the cognitive test.
correct = sum(1 for r in results if r["is_correct"])
print(f"\nAccuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
