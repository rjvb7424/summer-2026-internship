"""
experiment.py
=============

Drives the whole evaluation:

    for each model:
        for each trial:
            reset the world, then loop up to max_turns:
                render state -> build prompt -> ask model -> parse -> step
            record success + full per-turn transcript
            SAVE results.json          (crash-safe, after every trial)
        unload the model

Design choices that match how you like to run these:
  * results.json is rewritten atomically after every trial, so a crash mid-run
    never loses completed trials.
  * per-model resume: on restart, trials already in results.json are skipped.
  * a model that fails to load is recorded and skipped; the run continues.
  * all console output lives here (via logging), never inside model/env code.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import tempfile
from pathlib import Path

import observation as obs
from actions import ActionParser
from config import Config
from models import build_model
from prompt import PromptBuilder
from success import ObjectiveChecker
from world import CustomCrafterEnv

LOG = logging.getLogger("crafter_experiment.run")

TILE_PIXELS = 24  # size of each map tile in the saved PNG frames


class ExperimentRunner:
    """Runs every model over every trial and writes the results."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.checker = ObjectiveChecker(cfg.objective)
        self.parser = ActionParser(cfg.actions.strategy, cfg.actions.fallback)
        self.prompt_builder = PromptBuilder(cfg.prompt, self.checker.label)
        self.env = CustomCrafterEnv(cfg.world, seed=cfg.experiment.seed)
        self.renderer = obs.ImageRenderer(self.env.textures, TILE_PIXELS)
        self.results = self._load_or_init_results()

    # =========================================================================
    #  Top-level loop
    # =========================================================================
    def run(self) -> dict:
        self.cfg.run_dir.mkdir(parents=True, exist_ok=True)
        for spec in self.cfg.models:
            self._run_model(spec)
        LOG.info("Done. Results at %s", self.cfg.results_path)
        return self.results

    def _run_model(self, spec) -> None:
        record = self.results["models"].setdefault(
            spec.name, {"backend": spec.backend, "slug": spec.slug, "error": None, "trials": []}
        )
        done = len(record["trials"])
        total = self.cfg.experiment.num_trials
        if done >= total:
            LOG.info("[%s] already complete (%d/%d) - skipping.", spec.name, done, total)
            return

        LOG.info("[%s] loading (resuming at trial %d/%d)...", spec.name, done, total)
        model = build_model(spec, self.cfg.objective.target)
        try:
            model.load()
        except Exception as exc:  # e.g. out-of-memory on a huge model
            LOG.error("[%s] failed to load: %s", spec.name, exc)
            record["error"] = f"load failed: {exc}"
            self._save()
            return

        try:
            for trial in range(done, total):
                LOG.info("[%s] trial %d/%d ...", spec.name, trial + 1, total)
                result = self._run_trial(spec, model, trial)
                record["trials"].append(result)
                self._save()  # crash-safe: persist after every trial
                LOG.info(
                    "[%s] trial %d -> %s (%d turns)",
                    spec.name, trial + 1,
                    "SUCCESS" if result["success"] else "fail",
                    result["turns_used"],
                )
        finally:
            model.unload()

    # =========================================================================
    #  One trial
    # =========================================================================
    def _run_trial(self, spec, model, trial: int) -> dict:
        world_seed = self._world_seed(trial)
        self.env.set_world_seed(world_seed)
        self.env.reset()

        system_prompt, _ = self.prompt_builder.build(self.env)
        turns: list[dict] = []
        success = False
        success_turn = None

        for turn in range(self.cfg.experiment.max_turns):
            # State the model is about to see.
            _, user_prompt = self.prompt_builder.build(self.env)
            pre_pos = [int(v) for v in self.env.player.pos]
            pre_facing = obs.describe_facing(self.env.player)
            map_text = obs.render_text_map(self.env.world, self.env.player)
            frame_rel = self._save_frame(spec, trial, turn)

            # Decision.
            raw_text, think_seconds = model.generate(system_prompt, user_prompt)
            parsed = self.parser.parse(raw_text)

            # Advance the world.
            _, reward, done, info = self.env.step(parsed.index)
            success = self.checker.is_success(self.env)

            turns.append({
                "turn": turn,
                "player_pos": pre_pos,
                "facing": pre_facing,
                "map_text": map_text,
                "frame": frame_rel,
                "prompt": user_prompt,
                "raw_response": raw_text,
                "parsed_action": parsed.name,
                "action_index": parsed.index,
                "parse_ok": parsed.ok,
                "think_seconds": round(think_seconds, 4),
                "inventory": {k: int(v) for k, v in info["inventory"].items() if v > 0},
                "achievements_unlocked": sorted(
                    k for k, v in info["achievements"].items() if v > 0
                ),
                "reward": float(reward),
                "done": bool(done),
                "success": success,
            })

            if success:
                success_turn = turn
                break
            if done:  # player died
                break

        return {
            "trial": trial,
            "world_seed": world_seed,
            "success": success,
            "success_turn": success_turn,
            "turns_used": len(turns),
            "final_inventory": {
                k: int(v) for k, v in self.env.player.inventory.items() if v > 0
            },
            "final_achievements": sorted(
                k for k, v in self.env.player.achievements.items() if v > 0
            ),
            "turns": turns,
        }

    # =========================================================================
    #  Helpers
    # =========================================================================
    def _world_seed(self, trial: int) -> int:
        base = self.cfg.experiment.seed
        return base if self.cfg.experiment.same_world_each_trial else base + trial

    def _save_frame(self, spec, trial: int, turn: int) -> str | None:
        if not self.cfg.experiment.save_frames:
            return None
        rel = Path("frames") / spec.slug / f"trial_{trial}" / f"turn_{turn}.png"
        self.renderer.save(self.env.world, self.env.player, self.cfg.run_dir / rel)
        return str(rel)

    def _load_or_init_results(self) -> dict:
        if self.cfg.results_path.exists():
            LOG.info("Found existing results - will resume.")
            return json.loads(self.cfg.results_path.read_text())
        return {
            "experiment": self.cfg.experiment.name,
            "objective": {
                "type": self.cfg.objective.type,
                "target": self.cfg.objective.target,
                "label": self.checker.label,
            },
            "world_size": list(self.cfg.world.size),
            "max_turns": self.cfg.experiment.max_turns,
            "num_trials": self.cfg.experiment.num_trials,
            "created": _now(),
            "updated": _now(),
            "models": {},
        }

    def _save(self) -> None:
        """Atomically rewrite results.json so a crash can't corrupt it."""
        self.results["updated"] = _now()
        path = self.cfg.results_path
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump(self.results, fh, indent=2)
        os.replace(tmp, path)


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")
