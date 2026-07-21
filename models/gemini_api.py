"""
models/gemini_api.py
====================

An agent backed by Google's Gemini API (models such as ``gemini-2.5-flash``).
Uses the current ``google-genai`` SDK. The API key is read from the environment
(``GEMINI_API_KEY``, falling back to ``GOOGLE_API_KEY``), populated by ``.env``.

The ``google-genai`` package is imported lazily so the rest of the harness runs
without it installed. AI-facing: ``generate`` never prints or inputs.
"""

from __future__ import annotations

import os
import time

from models.base import LanguageModel
from models.conversation import ConversationMemory

DEFAULT_KEY_ENV = "GEMINI_API_KEY"
FALLBACK_KEY_ENV = "GOOGLE_API_KEY"


class GeminiModel(LanguageModel):
    """Calls a Gemini model once per turn."""

    def __init__(
        self,
        name: str,
        model: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        api_key_env: str = DEFAULT_KEY_ENV,
        history_turns: int = 0,
    ):
        super().__init__(name)
        self._model_id = model or name  # config 'name' doubles as the model id
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
        self._api_key_env = api_key_env
        self._memory = ConversationMemory(history_turns)
        self._client = None

    def reset(self) -> None:
        self._memory.reset()

    # -- lifecycle ------------------------------------------------------------
    def load(self) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "google-genai not installed - run: pip install google-genai"
            ) from exc

        api_key = os.environ.get(self._api_key_env) or os.environ.get(FALLBACK_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"No API key found. Set {self._api_key_env} (or {FALLBACK_KEY_ENV}) "
                f"in your .env file."
            )
        self._client = genai.Client(api_key=api_key)

    # -- inference ------------------------------------------------------------
    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, float]:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_prompt or None,
            max_output_tokens=self._max_tokens,
            temperature=self._temperature,
        )

        start = time.perf_counter()
        response = self._client.models.generate_content(
            model=self._model_id,
            contents=self._memory.gemini_contents(user_prompt),
            config=config,
        )
        elapsed = time.perf_counter() - start

        # response.text can be empty if the model returned no text part (e.g. a
        # safety block); fall back to an empty string so the turn still records.
        text = (getattr(response, "text", None) or "").strip()
        self._memory.record(user_prompt, text)
        return text, elapsed
