"""
models/openai_api.py
====================

An agent backed by the OpenAI Chat Completions API (ChatGPT models such as
``gpt-4o-mini``, ``gpt-4o``). The API key is read from the environment
(``OPENAI_API_KEY`` by default), which the ``.env`` file populates - it is
never placed in the config or in code.

The ``openai`` package is imported lazily so the rest of the harness runs
without it installed. AI-facing: ``generate`` never prints or inputs.
"""

from __future__ import annotations

import os
import time

from models.base import LanguageModel

DEFAULT_KEY_ENV = "OPENAI_API_KEY"


class OpenAIModel(LanguageModel):
    """Calls an OpenAI chat model once per turn."""

    def __init__(
        self,
        name: str,
        model: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        api_key_env: str = DEFAULT_KEY_ENV,
        base_url: str | None = None,
    ):
        super().__init__(name)
        self._model_id = model or name  # config 'name' doubles as the model id
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
        self._api_key_env = api_key_env
        self._base_url = base_url
        self._client = None

    # -- lifecycle ------------------------------------------------------------
    def load(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package not installed - run: pip install openai"
            ) from exc

        api_key = os.environ.get(self._api_key_env)
        if not api_key:
            raise RuntimeError(
                f"No API key found. Set {self._api_key_env} in your .env file."
            )
        self._client = OpenAI(api_key=api_key, base_url=self._base_url)

    # -- inference ------------------------------------------------------------
    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, float]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # max_completion_tokens is the current parameter and works for both chat
        # models (gpt-4o, ...) and reasoning models (o1, o3, ...). Older 'max_tokens'
        # is rejected by reasoning models. Reasoning models also reject a custom
        # temperature; the retry below drops any parameter the model complains about.
        params = {
            "model": self._model_id,
            "messages": messages,
            "max_completion_tokens": self._max_tokens,
            "temperature": self._temperature,
        }

        start = time.perf_counter()
        response = self._create(params)
        elapsed = time.perf_counter() - start

        text = (response.choices[0].message.content or "").strip()
        return text, elapsed

    def _create(self, params: dict):
        """Call the API, retrying without any parameter the model rejects
        (e.g. reasoning models refuse 'temperature')."""
        import openai

        for _ in range(4):
            try:
                return self._client.chat.completions.create(**params)
            except openai.BadRequestError as exc:
                bad = self._rejected_param(exc)
                if bad and bad in params and bad not in ("model", "messages"):
                    params.pop(bad)
                    continue
                raise

    @staticmethod
    def _rejected_param(exc) -> str | None:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            err = body.get("error", body)
            if isinstance(err, dict):
                return err.get("param")
        return None