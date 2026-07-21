"""
models/registry.py
==================

Factory that turns a :class:`config.ModelSpec` into a concrete
:class:`LanguageModel`.

Backends, each its own class:
  mock            -> MockModel            (no download; baselines)
  huggingface     -> HuggingFaceModel     (local transformers on your machine)
  openai          -> OpenAIModel          (OpenAI API: gpt-*, o*)
  gemini          -> GeminiModel          (Google Gemini API)
  huggingface-api -> HuggingFaceAPIModel  (HF models in the cloud)
"""

from __future__ import annotations

from config import ModelSpec
from models.base import LanguageModel
from models.mock import MockModel, GOAL_SYMBOL_BY_TARGET
from models.huggingface_local import HuggingFaceModel
from models.openai_api import OpenAIModel
from models.huggingface_api import HuggingFaceAPIModel
from models.gemini_api import GeminiModel


def _openai_style_kwargs(opts: dict) -> dict:
    """Shared options for the OpenAI-compatible backends (OpenAI + HF cloud)."""
    return dict(
        model=opts.get("model"),
        max_tokens=int(opts.get("max_tokens", opts.get("max_new_tokens", 256))),
        temperature=float(opts.get("temperature", 0.7)),
        request_delay=float(opts.get("request_delay", 0.0)),
        max_retries=int(opts.get("max_retries", 5)),
        retry_base_delay=float(opts.get("retry_base_delay", 2.0)),
        force_action=bool(opts.get("force_action", False)),
        action_retries=int(opts.get("action_retries", 0)),
        reasoning_effort=opts.get("reasoning_effort"),
        history_turns=int(opts.get("history_turns", 8)),
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
            history_turns=int(opts.get("history_turns", 8)),
        )

    if backend in ("openai", "chatgpt", "gpt"):
        return OpenAIModel(
            name=spec.name,
            api_key_env=opts.get("api_key_env", "OPENAI_API_KEY"),
            base_url=opts.get("base_url"),
            **_openai_style_kwargs(opts),
        )

    if backend in ("huggingface-api", "hf-api", "hf-cloud"):
        return HuggingFaceAPIModel(
            name=spec.name,
            api_key_env=opts.get("api_key_env", "HF_TOKEN"),
            base_url=opts.get("base_url", "https://router.huggingface.co/v1"),
            **_openai_style_kwargs(opts),
        )

    if backend in ("gemini", "google"):
        return GeminiModel(
            name=spec.name,
            model=opts.get("model"),
            max_tokens=int(opts.get("max_tokens", opts.get("max_new_tokens", 256))),
            temperature=float(opts.get("temperature", 0.7)),
            api_key_env=opts.get("api_key_env", "GEMINI_API_KEY"),
            history_turns=int(opts.get("history_turns", 8)),
        )

    raise ValueError(f"Unknown model backend '{spec.backend}' for '{spec.name}'.")