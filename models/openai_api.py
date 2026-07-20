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

import logging
import os
import random
import time

from models.base import LanguageModel

LOG = logging.getLogger("crafter_experiment.models.openai")
DEFAULT_KEY_ENV = "OPENAI_API_KEY"
MAX_BACKOFF = 60.0  # cap a single wait at one minute


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
        request_delay: float = 0.0,
        max_retries: int = 5,
        retry_base_delay: float = 2.0,
    ):
        super().__init__(name)
        self._model_id = model or name  # config 'name' doubles as the model id
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
        self._api_key_env = api_key_env
        self._base_url = base_url
        self._request_delay = float(request_delay)      # throttle between calls
        self._max_retries = int(max_retries)            # retries on rate limit
        self._retry_base_delay = float(retry_base_delay)  # backoff base seconds
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

        # Throttle: space out requests to stay under rate limits. Not counted as
        # think time, so it won't distort the latency numbers.
        if self._request_delay > 0:
            time.sleep(self._request_delay)

        return self._create(params)

    def _create(self, params: dict) -> tuple[str, float]:
        """Call the API. Drops any parameter the model rejects, and on a rate
        limit waits (honouring Retry-After) and retries instead of crashing."""
        import openai

        rate_limit_attempts = 0
        while True:
            try:
                start = time.perf_counter()
                response = self._client.chat.completions.create(**params)
                elapsed = time.perf_counter() - start
                text = (response.choices[0].message.content or "").strip()
                return text, elapsed
            except openai.BadRequestError as exc:
                bad = self._rejected_param(exc)
                if bad and bad in params and bad not in ("model", "messages"):
                    params.pop(bad)
                    continue
                raise
            except (openai.RateLimitError, openai.APITimeoutError,
                    openai.APIConnectionError) as exc:
                rate_limit_attempts += 1
                if rate_limit_attempts > self._max_retries:
                    raise
                wait = self._retry_wait(exc, rate_limit_attempts)
                LOG.warning(
                    "[%s] rate limited / transient error - waiting %.1fs, retry %d/%d",
                    self.name, wait, rate_limit_attempts, self._max_retries,
                )
                time.sleep(wait)

    def _retry_wait(self, exc, attempt: int) -> float:
        """Seconds to wait: the server's Retry-After if given, else exponential
        backoff with a little jitter."""
        retry_after = self._retry_after(exc)
        if retry_after is not None:
            return retry_after
        backoff = self._retry_base_delay * (2 ** (attempt - 1))
        return min(backoff, MAX_BACKOFF) + random.uniform(0, 0.5)

    @staticmethod
    def _retry_after(exc) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            value = headers.get("retry-after")
            if value:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
        return None

    @staticmethod
    def _rejected_param(exc) -> str | None:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            err = body.get("error", body)
            if isinstance(err, dict):
                return err.get("param")
        return None
