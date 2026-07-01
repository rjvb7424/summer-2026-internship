import json
import os
import paper
import gemini

NUM_TRIALS = 20
RESULTS_FILE = "results.json"

def load_existing_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    return []

def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

def gemini_solver(prompt):
    return gemini.call_gemini(prompt)

results = load_existing_results()
start_trial = len(results)

for i in range(start_trial, NUM_TRIALS):
    print(f"\n=== Trial {i+1}/{NUM_TRIALS} ===")
    test = paper.CognitiveTest()
    result = test.run_automated(num_folds=3, solver=gemini_solver)
    result["trial"] = i + 1
    results.append(result)

    save_results(results)  # write after every trial, not just at the end

    print(f"Correct: {result['correct_choice']} | Gemini: {result['predicted_choice']} | Match: {result['is_correct']}")

correct = sum(1 for r in results if r["is_correct"])
print(f"\nAccuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
