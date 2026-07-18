# Crafter LLM Experiment Harness

Drop a language model into a hand-built Crafter world, give it a snapshot each
turn, let it pick an action, and measure whether it completes an objective using
Crafter's **built-in achievement system**. Everything — world layout, goal,
models, turn/trial counts — is driven by one YAML file.

```
python main.py                 # run + plots + viewer, using config.yaml
python main.py my_config.yaml  # a different experiment
```

Outputs land in `runs/<experiment_name>/`:

```
results.json          every turn of every trial (saved after each trial)
videos/               ONE mp4 per model - the full run, start to finish
plots/                success_rate, turns_to_success, think_time, success_matrix
viewer.html           interactive replay: state + prompt + response + timing
```

---

## Install

```
pip install -r requirements.txt
```

`torch`, `transformers`, `accelerate` are only needed for the `huggingface`
backend (local models). The `mock` baselines run with nothing but Crafter +
numpy + Pillow. On Apple Silicon, `torch` gives you the MPS device automatically.

---

## The one loop

Each turn the harness:

1. Renders the world to a text map (and an in-memory frame for the video).
2. Fills your prompt template with the map, legend, inventory, achievements,
   position and facing.
3. Sends it to the model and times the response.
4. Parses an action out of the reply and steps the environment.
5. Checks the objective. Stops the trial on success (or death, or `max_turns`).

---

## Customising — it's all in the config

### Change the world size

```yaml
world:
  size: [15, 15]
```

### Place things where you want them

`positions` puts features at exact tiles; `count` scatters them randomly on free
ground; `rect: [x, y, w, h]` fills a block (used here for the pond).

```yaml
world:
  features:
    trees:
      positions: [[2, 2], [7, 3], [4, 7]]
    water:
      rect: [6, 6, 3, 3]
    stone:
      count: 4
  entities:
    cow:
      positions: [[3, 5]]
```

### Swap the objective (the whole point)

Change **one line**. Any of Crafter's 22 achievements works:

```yaml
objective:
  type: achievement
  target: make_stone_pickaxe      # was: collect_wood
```

Or check inventory quantities instead:

```yaml
objective:
  type: inventory
  item: wood
  amount: 3
```

The success test lives in `success.py` (`ObjectiveChecker`) — that's the single
module to edit if you ever want a bespoke win condition.

### Turns, trials, seeds

```yaml
experiment:
  num_trials: 5
  max_turns: 100
  seed: 0
  same_world_each_trial: true   # false = a fresh seeded layout per trial
```

### Models

```yaml
models:
  - name: Qwen/Qwen2.5-3B-Instruct
    backend: huggingface
    max_new_tokens: 256
    temperature: 0.7
  - name: heuristic-baseline      # zero-download sanity check
    backend: mock
    policy: heuristic
```

`huggingface` models load **one at a time** and are unloaded (with cache
clearing) before the next — so a big model won't sit in memory alongside the
next one. Runs are resumable: if a run is interrupted, rerunning skips the
trials already in `results.json`.

---

## The video

At the end of a run, each model's turns are stitched into a single MP4 at
`runs/<name>/videos/<model>.mp4` - one continuous video of the whole run, with a
title card before each trial. No per-turn screenshots are written to disk.

Turn it off or change the speed in the config:

```yaml
experiment:
  record_video: true
  video_fps: 4
```

Needs `imageio-ffmpeg` (in requirements.txt) - it bundles ffmpeg, so there is no
system install.

## Watching it run live

```
python main.py --live            # runs + serves a real-time view
```

This starts a small local web server (default http://127.0.0.1:8000) and prints
the URL. Open it in a browser and the page refreshes itself every turn while the
run happens: current game state, the prompt, the model's raw response, the
action taken, the think time, and a running solved-count. Pick a different port
with `--port 8123`. When the run finishes the final state stays on screen until
you press Ctrl-C.

Use this to confirm an experiment is actually working. For a careful after-the-
fact review, use the static replay below.

## Seeing what the model did

```
python viewer.py           # (re)build viewer.html from the latest run
```

Open `runs/<name>/viewer.html`. Pick a model and trial, then scrub through the
turns (arrow keys work). Each turn shows the rendered state, the exact prompt,
the model's raw response, the action taken, the turn number and the think time.

Rebuild plots without rerunning trials:

```
python main.py --skip-run
```

---

## File map

| File | Role |
|---|---|
| `config.yaml` / `config.py` | the experiment definition + typed loader/validator |
| `world.py` | custom Crafter env + world builder |
| `observation.py` | world → text map, legend, PNG frames |
| `prompt.py` | fills the prompt template |
| `actions.py` | parses an action out of the model's text |
| `success.py` | **the swappable objective checker** |
| `models/` | model backends (`huggingface_local`, `mock`) + registry |
| `experiment.py` | the runner: trials, logging, crash-safe saving, resume |
| `analyze_results.py` | results.json → plots |
| `viewer.py` | results.json → viewer.html |
| `videos.py` | in-memory frames → one mp4 per model |
| `main.py` | run + analyze + viewer in one command |

---

## Notes

- Local HuggingFace inference needs network access on first load (to download
  weights) and, for gated models, a logged-in `huggingface-cli`.
- All model-facing code (`world`, `observation`, `prompt`, `actions`,
  `success`, `models/`) is free of `print`/`input`; console output goes through
  `logging` in the runner only.
