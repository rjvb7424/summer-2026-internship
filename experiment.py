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
import io
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import observation as obs
import videos
from actions import ActionParser
from config import Config
from live_viewer import DEFAULT_PORT, LiveViewer
from models import build_model
from prompt import PromptBuilder
from success import ObjectiveChecker
from world import CustomCrafterEnv

LOG = logging.getLogger("crafter_experiment.run")


class ExperimentRunner:
    """Runs every model over every trial and writes the results."""

    def __init__(
        self, cfg: Config, live: bool = False,
        live_port: int = DEFAULT_PORT, open_browser: bool = True,
    ):
        self.cfg = cfg
        self.checker = ObjectiveChecker(cfg.objective)
        self.parser = ActionParser(cfg.actions.strategy, cfg.actions.fallback)
        self.prompt_builder = PromptBuilder(cfg.prompt, self.checker.label)
        self.env = CustomCrafterEnv(cfg.world, seed=cfg.experiment.seed)
        self._tile_px = self._tile_pixels()
        self.renderer = obs.ImageRenderer(self.env.textures, self._tile_px)
        self.results = self._load_or_init_results()

        # Optional real-time browser view, updated once per turn during the run.
        self.live: LiveViewer | None = None
        if live:
            self.live = LiveViewer(
                self.cfg.run_dir, self.cfg.experiment.name, self.checker.label,
                port=live_port,
            )
            self.cfg.run_dir.mkdir(parents=True, exist_ok=True)
            self.live.start(open_browser=open_browser)

    # =========================================================================
    #  Top-level loop
    # =========================================================================
    def run(self) -> dict:
        self.cfg.run_dir.mkdir(parents=True, exist_ok=True)
        for spec in self.cfg.models:
            self._run_model(spec)
        if self.live:
            self.live.set_complete()
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

        successes = sum(1 for t in record["trials"] if t["success"])
        writer = self._new_video_writer(spec)
        try:
            for trial in range(done, total):
                LOG.info("[%s] trial %d/%d ...", spec.name, trial + 1, total)
                result = self._run_trial(spec, model, trial, successes, trial, writer)
                record["trials"].append(result)
                successes += int(result["success"])
                self._save()  # crash-safe: persist after every trial
                LOG.info(
                    "[%s] trial %d -> %s (%d turns)",
                    spec.name, trial + 1,
                    "SUCCESS" if result["success"] else "fail",
                    result["turns_used"],
                )
        finally:
            model.unload()

        if writer is not None:
            path = writer.close()
            if path:
                LOG.info("[%s] video: %s", spec.name, path)

    # =========================================================================
    #  Rendering / video helpers
    # =========================================================================
    def _tile_pixels(self) -> int:
        """Pixels per tile, chosen so the video's longest side is about
        `video_resolution`. Snapped to a multiple of 16 (the source art is
        16x16) so pixel-art upscaling stays crisp."""
        res = int(self.cfg.experiment.video_resolution)
        longest = max(self.cfg.world.size)
        return max(16, round(res / longest / 16) * 16)

    def _new_video_writer(self, spec):
        if not self.cfg.experiment.record_video:
            return None
        width, height = self.cfg.world.size
        size = (width * self._tile_px, height * self._tile_px)
        out = self.cfg.videos_dir / f"{spec.slug}.mp4"
        return videos.VideoWriter(out, self.cfg.experiment.video_fps, size)

    # =========================================================================
    #  One trial
    # =========================================================================
    def _run_trial(
        self, spec, model, trial: int,
        prior_successes: int = 0, prior_trials: int = 0, video_writer=None,
    ) -> dict:
        world_seed = self._world_seed(trial)
        self.env.set_world_seed(world_seed)
        self.env.reset()
        model.reset()  # clear conversation memory so trials don't bleed together

        system_prompt, _ = self.prompt_builder.build(self.env)
        turns: list[dict] = []
        need_frames = self.live is not None or video_writer is not None
        last_frame = None
        success = False
        success_turn = None

        if video_writer is not None:
            video_writer.title([spec.name, f"trial {trial + 1}"])

        for turn in range(self.cfg.experiment.max_turns):
            # State the model is about to see.
            _, user_prompt = self.prompt_builder.build(self.env)
            pre_pos = [int(v) for v in self.env.player.pos]
            pre_facing = obs.describe_facing(self.env.player)
            map_text = obs.render_text_map(self.env.world, self.env.player)

            # Frame is rendered in memory: streamed to the video and/or shown in
            # the live view. No per-turn PNG is written to disk.
            frame_url = None
            if need_frames:
                frame_arr = self.renderer.render_array(self.env.world, self.env.player)
                last_frame = frame_arr
                if video_writer is not None:
                    video_writer.frame(frame_arr)
                if self.live is not None:
                    self.live.set_frame(self._png_bytes(frame_arr))
                    frame_url = "frame.png"

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
                "frame": None,
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

            if self.live:
                self.live.update({
                    "model": spec.name,
                    "backend": spec.backend,
                    "trial": trial + 1,
                    "num_trials": self.cfg.experiment.num_trials,
                    "turn": turn + 1,
                    "max_turns": self.cfg.experiment.max_turns,
                    "action": parsed.name,
                    "parse_ok": parsed.ok,
                    "think_seconds": round(think_seconds, 3),
                    "facing": pre_facing,
                    "map_text": map_text,
                    "frame": frame_url,
                    "prompt": user_prompt,
                    "raw_response": raw_text,
                    "inventory": {k: int(v) for k, v in info["inventory"].items() if v > 0},
                    "achievements": sorted(
                        k for k, v in info["achievements"].items() if v > 0
                    ),
                    "success": success,
                    "successes": prior_successes + int(success),
                    "trials_done": prior_trials + 1,
                })

            if success:
                success_turn = turn
                break
            if done:  # player died
                break

        # Close out the trial in the video: hold the final frame, then a short
        # outcome card.
        if video_writer is not None and last_frame is not None:
            video_writer.hold(last_frame)
            video_writer.title(
                [f"trial {trial + 1}", "solved" if success else "not solved"],
                hold=4,
            )

        return {
            "trial": trial + 1,
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

    def _png_bytes(self, frame_arr: np.ndarray) -> bytes:
        """Encode a frame array to PNG bytes for the live view (no disk)."""
        buf = io.BytesIO()
        Image.fromarray(frame_arr).save(buf, format="PNG")
        return buf.getvalue()

    def _config_fingerprint(self) -> dict:
        """The settings that make trials comparable. If any of these change, old
        trials in results.json are no longer valid to merge with new ones."""
        return {
            "objective": {
                "type": self.cfg.objective.type,
                "target": self.cfg.objective.target,
                "item": self.cfg.objective.item,
                "amount": self.cfg.objective.amount,
            },
            "world_size": list(self.cfg.world.size),
            "max_turns": self.cfg.experiment.max_turns,
        }

    def _load_or_init_results(self) -> dict:
        fingerprint = self._config_fingerprint()
        if self.cfg.results_path.exists():
            existing = json.loads(self.cfg.results_path.read_text())
            old_fp = existing.get("config_fingerprint")
            if old_fp is not None and old_fp != fingerprint:
                raise SystemExit(
                    f"\n{self.cfg.results_path} was produced with a DIFFERENT "
                    f"configuration (objective, world size or max_turns changed).\n"
                    f"Merging old and new trials would give meaningless graphs.\n"
                    f"Fix: either rename 'experiment.name' in your config, or delete "
                    f"the folder  {self.cfg.run_dir}  and run again.\n"
                    f"  old: {old_fp}\n  new: {fingerprint}\n"
                )
            LOG.info("Found existing results - resuming (config matches).")
            existing.setdefault("config_fingerprint", fingerprint)
            return existing
        return {
            "experiment": self.cfg.experiment.name,
            "config_fingerprint": fingerprint,
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
