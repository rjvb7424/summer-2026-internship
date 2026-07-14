# Crafter Hugging Face Wood Experiment

A fixed 9x9 Crafter world where a Hugging Face model receives a text map,
chooses one action per turn, and tries to unlock `collect_wood`.

## Files

- `config.py`: model, map, goal, and trial settings
- `crafter_env.py`: fixed-world construction and text observations
- `hf_agent.py`: Hugging Face model loading, generation, and action parsing
- `main.py`: trials, turns, visualization, stopping, and results
- `requirements.txt`: dependencies

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Create another task

Change the world in `config.py`:

```python
PLAYER_POSITION = (4, 4)

TILES = {
    (4, 2): "tree",
    (2, 3): "tree",
}
```

Change the experiment:

```python
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
NUM_TRIALS = 10
MAX_TURNS = 20
GOAL_ACHIEVEMENT = "collect_wood"
GOAL_DESCRIPTION = "Collect wood from one tree."
```

Each trial is saved to `wood_results.json`, including every raw model
response, parsed action, reward, position, facing direction, and inventory.
