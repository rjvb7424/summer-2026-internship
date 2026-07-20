"""
models/registry.py
==================

Factory that turns a :class:`config.ModelSpec` into a concrete
:class:`LanguageModel`.
"""

from __future__ import annotations

from config import ModelSpec
from models.base import LanguageModel
from models.mock import MockModel, GOAL_SYMBOL_BY_TARGET
from models.huggingface_local import HuggingFaceModel
from models.openai_api import OpenAIModel
from models.gemini_api import GeminiModel


def _openai_compatible_model(spec: ModelSpec, *, api_key_env: str, base_url: str | None):
    opts = spec.options
    return OpenAIModel(
        name=spec.name,
        model=opts.get("model"),
        max_tokens=int(opts.get("max_tokens", opts.get("max_new_tokens", 256))),
        temperature=float(opts.get("temperature", 0.7)),
        api_key_env=opts.get("api_key_env", api_key_env),
        base_url=opts.get("base_url", base_url),
        request_delay=float(opts.get("request_delay", 0.0)),
        max_retries=int(opts.get("max_retries", 5)),
        retry_base_delay=float(opts.get("retry_base_delay", 2.0)),
        force_action=bool(opts.get("force_action", False)),
        action_retries=int(opts.get("action_retries", 0)),
        reasoning_effort=opts.get("reasoning_effort"),
    )


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
            token_env=opts.get("hf_token_env"),
        )

    if backend in ("openai", "chatgpt", "gpt"):
        return _openai_compatible_model(
            spec,
            api_key_env="OPENAI_API_KEY",
            base_url=None,
        )

    if backend in ("gemini", "google"):
        return GeminiModel(
            name=spec.name,
            model=opts.get("model"),
            max_tokens=int(opts.get("max_tokens", opts.get("max_new_tokens", 256))),
            temperature=float(opts.get("temperature", 0.7)),
            api_key_env=opts.get("api_key_env", "GEMINI_API_KEY"),
        )

    if backend in ("huggingface-api", "hf-api", "hf-cloud"):
        return _openai_compatible_model(
            spec,
            api_key_env="HF_TOKEN",
            base_url="https://router.huggingface.co/v1",
        )

    raise ValueError(f"Unknown model backend '{spec.backend}' for '{spec.name}'.")