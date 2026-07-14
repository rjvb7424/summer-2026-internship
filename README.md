# Crafter LLM Spatial Reasoning Benchmark

Benchmarks HuggingFace models on the Crafter survival environment with a live
viewer, per-trial video recordings, crash-safe JSON results, and automatic
graph generation.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## API keys (only for API providers)

```bash
export OPENAI_API_KEY=...    # for ("openai", ...) models
export GEMINI_API_KEY=...    # for ("gemini", ...) models
```

## Run

```bash
python main.py
```

Everything is configured in `config.py` (models, trials, steps, viewer,
recording, seeds). Runs are resumable: results are written after every trial,
so interrupting and rerunning `main.py` continues from the last completed
trial. Delete `results/<model>.json` to redo a model.

Outputs:
- `results/<model>.json` — one record per trial (reward, steps, achievements, actions, invalid-action count)
- `recordings/<model>/trial_NNN.mp4` — video of every trial
- `results/graphs/*.png` — comparison graphs (also regenerable via `python analyze_results.py`)

## Structure

| File | Role |
|---|---|
| `config.py` | all experiment flags |
| `main.py` | experiment loop |
| `crafter_env.py` | Crafter wrapper + text observations |
| `base_agent.py` | shared prompt/history/action extraction |
| `hf_agent.py` | local HuggingFace models (MPS) |
| `openai_agent.py` | ChatGPT API models |
| `gemini_agent.py` | Gemini API models |
| `viewer.py` | live pygame window |
| `recorder.py` | per-trial mp4 recording |
| `results_store.py` | crash-safe JSON persistence |
| `analyze_results.py` | graphs |

## Apple Silicon notes

- Models run on MPS in fp16 automatically; `MPS_HIGH_WATERMARK_RATIO` in
  `config.py` caps GPU memory at 85% of unified memory so a too-large model
  errors instead of freezing the machine.
- One model is loaded at a time and fully unloaded (`gc` + MPS cache flush)
  before the next.
- Rough fp16 sizing on a 18 GB M3 Pro: ≤7B params is comfortable, ~8B is the
  practical ceiling. For anything larger, prefer an MLX 4-bit variant instead.
