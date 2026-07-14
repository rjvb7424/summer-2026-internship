"""
models/registry.py
==================

Factory that turns a :class:`config.ModelSpec` into a concrete
:class:`LanguageModel`. Add a new backend by extending ``build_model``.
"""

from __future__ import annotations

from config import ModelSpec
from models.base import LanguageModel
from models.mock import MockModel, GOAL_SYMBOL_BY_TARGET
from models.huggingface_local import HuggingFaceModel


def build_model(spec: ModelSpec, objective_target: str | None = None) -> LanguageModel:
    """Instantiate the agent described by ``spec``."""
    opts = spec.options
    backend = spec.backend.lower()

    if backend == "mock":
        goal_symbol = opts.get("goal_symbol") or GOAL_SYMBOL_BY_TARGET.get(objective_target or "")
        return MockModel(
            name=spec.name,
            policy=opts.get("policy", "heuristic"),
            fixed_action=opts.get("fixed_action", "noop"),
            goal_symbol=goal_symbol,
        )

    if backend == "huggingface":
        return HuggingFaceModel(
            name=spec.name,
            max_new_tokens=int(opts.get("max_new_tokens", 256)),
            temperature=float(opts.get("temperature", 0.7)),
            dtype=opts.get("dtype", "auto"),
            device=opts.get("device", "auto"),
        )

    raise ValueError(f"Unknown model backend '{spec.backend}' for '{spec.name}'.")
