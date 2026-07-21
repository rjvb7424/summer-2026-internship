"""
models/base.py
==============

The interface every agent implements. The experiment runner only ever calls:

    model.load()                     # bring weights into memory (may be no-op)
    raw, secs = model.generate(...)  # one decision
    model.unload()                   # free memory before the next model

Keeping this tiny means dropping in an OpenAI/Gemini/other backend later is
just another subclass. AI-facing: implementations must not print or input.
"""

from __future__ import annotations

import abc


class LanguageModel(abc.ABC):
    """Base class for all agents evaluated by the experiment."""

    def __init__(self, name: str):
        self.name = name

    def load(self) -> None:
        """Load weights / open clients. Default: nothing to do."""

    def unload(self) -> None:
        """Release memory. Default: nothing to do."""

    def reset(self) -> None:
        """Clear any per-episode state (e.g. conversation memory). Called by the
        runner at the start of each trial. Default: nothing to do."""

    @abc.abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, float]:
        """
        Produce one reply.

        Returns
        -------
        (raw_text, think_seconds)
            ``raw_text`` is the model's unedited output; ``think_seconds`` is
            the wall-clock time spent generating it.
        """
