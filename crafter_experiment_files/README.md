# Prompt-Based AI Models in Crafter

This adds a Crafter experiment beside the existing `gemini.py`, `gpt.py`, and
`huggingface.py` solver modules.

## Install

A Python 3.11 or 3.12 virtual environment is recommended because Crafter 1.8.3
still uses the older Gym-style API.

```bash
python -m pip install -r requirements-crafter.txt
```

Keep your provider dependencies and API keys configured as before.

## Run

```bash
python run_crafter.py
```

The Pygame window remains responsive while the model is loading or generating.
Press Escape or close the window to stop cleanly. Results are saved after every
episode in `crafter_results.json`. When `RECORD_VIDEO = True`, Crafter also
writes episode recordings under `crafter_recordings/`.

## Important experiment choice

Crafter's native observation is a 64x64 RGB image. Your current solver contract
accepts only a text prompt, so `CrafterTest` converts the visible local map,
player direction, inventory, rewards, and achievements into a shared symbolic
text observation. This allows text-only and multimodal-capable chat models to be
compared through the same interface, but it is a symbolic-observation Crafter
experiment rather than the standard pixel-only benchmark.

## Fair comparison

Each model receives the same seed for the same trial number:

```python
seed = BASE_SEED + trial_index
```

Do not change the prompt, maximum steps, generation settings, or seed schedule
between models if you want a fair comparison.

## Main settings

Edit these constants in `run_crafter.py`:

- `NUM_TRIALS`
- `MAX_STEPS`
- `SHOW_SIMULATION`
- `RECORD_VIDEO`
- provider toggles and model lists

The default `MAX_STEPS = 100` means up to 100 model calls per episode. Increase
it only after confirming the latency and cost are acceptable.
