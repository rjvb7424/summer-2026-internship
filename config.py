"""
config.py
=========

Loads ``config.yaml`` into typed, defaulted dataclasses and validates it against
the *installed* Crafter package (so the set of legal achievements/actions always
matches whatever Crafter version is present).

Nothing in here is AI-facing; it is pure setup/validation and may log warnings.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ruamel.yaml
import crafter.constants as C

LOG = logging.getLogger("crafter_experiment.config")

# --- Ground truth pulled straight from the installed Crafter -----------------
VALID_ACHIEVEMENTS: tuple[str, ...] = tuple(C.achievements)
VALID_ACTIONS: tuple[str, ...] = tuple(C.actions)
VALID_ITEMS: tuple[str, ...] = tuple(C.items.keys())
VALID_MATERIALS: tuple[str, ...] = tuple(C.materials)


# =============================================================================
#  Section dataclasses
# =============================================================================
@dataclass
class ExperimentCfg:
    """Top-level run budgets and output settings."""

    name: str = "experiment"
    output_dir: str = "runs"
    num_trials: int = 5
    max_turns: int = 100
    seed: int = 0
    same_world_each_trial: bool = True
    record_video: bool = True
    video_fps: int = 4

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentCfg":
        return cls(
            name=d.get("name", "experiment"),
            output_dir=d.get("output_dir", "runs"),
            num_trials=int(d.get("num_trials", 5)),
            max_turns=int(d.get("max_turns", 100)),
            seed=int(d.get("seed", 0)),
            same_world_each_trial=bool(d.get("same_world_each_trial", True)),
            record_video=bool(d.get("record_video", True)),
            video_fps=int(d.get("video_fps", 4)),
        )


@dataclass
class ObjectiveCfg:
    """Defines the single success condition for the experiment."""

    type: str = "achievement"          # 'achievement' | 'inventory'
    target: str = "collect_wood"       # achievement name
    item: str = "wood"                 # item name (inventory objective)
    amount: int = 1                    # required count (inventory objective)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ObjectiveCfg":
        return cls(
            type=d.get("type", "achievement"),
            target=d.get("target", "collect_wood"),
            item=d.get("item", "wood"),
            amount=int(d.get("amount", 1)),
        )

    @property
    def label(self) -> str:
        """Human-readable goal string injected into the prompt."""
        if self.type == "inventory":
            return f"collect at least {self.amount} {self.item}"
        return self.target.replace("_", " ")


@dataclass
class WorldCfg:
    """Everything about the physical world. Feature/entity specs stay as dicts."""

    size: tuple[int, int] = (10, 10)
    base_terrain: str = "grass"
    player_start: tuple[int, int] | None = None
    static: bool = True
    freeze_daylight: bool = True
    inventory: dict[str, int] = field(default_factory=dict)
    features: dict[str, dict] = field(default_factory=dict)
    entities: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorldCfg":
        size = tuple(d.get("size", [10, 10]))
        start = d.get("player_start", None)
        return cls(
            size=(int(size[0]), int(size[1])),
            base_terrain=d.get("base_terrain", "grass"),
            player_start=tuple(start) if start else None,
            static=bool(d.get("static", True)),
            freeze_daylight=bool(d.get("freeze_daylight", True)),
            inventory=dict(d.get("inventory", {}) or {}),
            features=dict(d.get("features", {}) or {}),
            entities=dict(d.get("entities", {}) or {}),
        )


@dataclass
class PromptCfg:
    """Prompt templates and which context blocks to include."""

    include_legend: bool = True
    include_inventory: bool = True
    include_achievements: bool = True
    include_action_list: bool = True
    system: str = ""
    user: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PromptCfg":
        return cls(
            include_legend=bool(d.get("include_legend", True)),
            include_inventory=bool(d.get("include_inventory", True)),
            include_achievements=bool(d.get("include_achievements", True)),
            include_action_list=bool(d.get("include_action_list", True)),
            system=d.get("system", "").strip(),
            user=d.get("user", "").strip(),
        )


@dataclass
class ActionParseCfg:
    """How raw model text is turned into an action."""

    strategy: str = "keyword"
    fallback: str = "noop"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ActionParseCfg":
        return cls(
            strategy=d.get("strategy", "keyword"),
            fallback=d.get("fallback", "noop"),
        )


@dataclass
class ModelSpec:
    """One model to evaluate. Extra keys are kept in ``options``."""

    name: str
    backend: str = "huggingface"
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelSpec":
        d = dict(d)
        name = d.pop("name")
        backend = d.pop("backend", "huggingface")
        return cls(name=name, backend=backend, options=d)

    @property
    def slug(self) -> str:
        """Filesystem-safe identifier used for frame folders."""
        return self.name.replace("/", "__").replace(" ", "_")


@dataclass
class Config:
    """The whole configuration, assembled from the sections above."""

    experiment: ExperimentCfg
    objective: ObjectiveCfg
    world: WorldCfg
    prompt: PromptCfg
    actions: ActionParseCfg
    models: list[ModelSpec]
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return Path(self.experiment.output_dir) / self.experiment.name

    @property
    def results_path(self) -> Path:
        return self.run_dir / "results.json"

    @property
    def videos_dir(self) -> Path:
        return self.run_dir / "videos"

    @property
    def plots_dir(self) -> Path:
        return self.run_dir / "plots"


# =============================================================================
#  Loading and validation
# =============================================================================
def load_config(path: str | Path) -> Config:
    """Read a YAML file and return a validated :class:`Config`."""
    yaml = ruamel.yaml.YAML(typ="safe", pure=True)
    raw = yaml.load(Path(path).read_text())

    cfg = Config(
        experiment=ExperimentCfg.from_dict(raw.get("experiment", {})),
        objective=ObjectiveCfg.from_dict(raw.get("objective", {})),
        world=WorldCfg.from_dict(raw.get("world", {})),
        prompt=PromptCfg.from_dict(raw.get("prompt", {})),
        actions=ActionParseCfg.from_dict(raw.get("actions", {})),
        models=[ModelSpec.from_dict(m) for m in raw.get("models", [])],
        raw=copy.deepcopy(raw),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    """Fail loudly on impossible configs; warn on suspicious ones."""
    # Objective must reference something real.
    if cfg.objective.type == "achievement":
        if cfg.objective.target not in VALID_ACHIEVEMENTS:
            raise ValueError(
                f"Unknown achievement '{cfg.objective.target}'.\n"
                f"Valid achievements: {', '.join(VALID_ACHIEVEMENTS)}"
            )
    elif cfg.objective.type == "inventory":
        if cfg.objective.item not in VALID_ITEMS:
            raise ValueError(
                f"Unknown item '{cfg.objective.item}'.\n"
                f"Valid items: {', '.join(VALID_ITEMS)}"
            )
    else:
        raise ValueError(f"objective.type must be 'achievement' or 'inventory'.")

    # World must be non-trivial.
    w, h = cfg.world.size
    if w < 3 or h < 3:
        raise ValueError(f"world.size {cfg.world.size} is too small (min 3x3).")

    # Fallback action must be legal.
    if cfg.actions.fallback not in VALID_ACTIONS:
        raise ValueError(
            f"actions.fallback '{cfg.actions.fallback}' is not a Crafter action."
        )

    # Starting inventory keys must be real items.
    for item in cfg.world.inventory:
        if item not in VALID_ITEMS:
            raise ValueError(f"world.inventory has unknown item '{item}'.")

    # At least one model.
    if not cfg.models:
        raise ValueError("No models listed under 'models:'.")

    # Warn (don't fail) on out-of-bounds explicit placements.
    for group in (cfg.world.features, cfg.world.entities):
        for kind, spec in group.items():
            for pos in (spec or {}).get("positions", []) or []:
                x, y = pos
                if not (0 <= x < w and 0 <= y < h):
                    LOG.warning("%s at %s is outside the %sx%s world.", kind, pos, w, h)
