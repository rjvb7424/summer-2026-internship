"""
models/openai_api.py
====================

OpenAI-compatible model backend used for OpenAI and Hugging Face Inference
Providers.

Reasoning models may spend their entire output budget on hidden/visible thinking
and never produce a final text answer.  When ``force_action`` is enabled this
backend forces a ``choose_action`` tool call whose argument is constrained to a
legal Crafter action.  The model can still reason internally, but the value
returned to the experiment is only the selected action name.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time

import crafter.constants as C

from models.base import LanguageModel

LOG = logging.getLogger("crafter_experiment.models.openai")
DEFAULT_KEY_ENV = "OPENAI_API_KEY"
MAX_BACKOFF = 60.0
MAX_ACTION_TOKEN_BUDGET = 8192

ACTIONS: tuple[str, ...] = tuple(C.actions)
_ACTION_RE = re.compile(
    rf"\b({'|'.join(re.escape(a) for a in sorted(ACTIONS, key=len, reverse=True))})\b",
    re.IGNORECASE,
)


class OpenAIModel(LanguageModel):
    """Calls an OpenAI-compatible chat-completions endpoint once per turn."""

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
        force_action: bool = False,
        action_retries: int = 0,
        reasoning_effort: str | None = None,
    ):
        super().__init__(name)
        self._model_id = model or name
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
        self._api_key_env = api_key_env
        self._base_url = base_url
        self._request_delay = float(request_delay)
        self._max_retries = int(max_retries)
        self._retry_base_delay = float(retry_base_delay)
        self._force_action = bool(force_action)
        self._action_retries = max(0, int(action_retries))
        self._reasoning_effort = reasoning_effort
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
        base_messages: list[dict] = []
        if system_prompt:
            base_messages.append({"role": "system", "content": system_prompt})
        base_messages.append({"role": "user", "content": user_prompt})

        if self._request_delay > 0:
            time.sleep(self._request_delay)

        attempts = 1 + (self._action_retries if self._force_action else 0)
        elapsed_total = 0.0
        last_response = None

        for attempt in range(attempts):
            messages = list(base_messages)
            if attempt:
                messages.append({
                    "role": "user",
                    "content": (
                        "The previous attempt did not return a legal action. "
                        "Choose exactly one action now by calling choose_action."
                    ),
                })

            # A reasoning model's completion budget includes its reasoning tokens.
            # On a recovery attempt, give it extra room to reach the tool call.
            token_budget = min(
                self._max_tokens * (2 ** attempt),
                MAX_ACTION_TOKEN_BUDGET,
            )
            params = self._request_params(messages, token_budget)
            response, elapsed = self._create_response(params)
            elapsed_total += elapsed
            last_response = response

            if not self._force_action:
                return self._response_text(response), elapsed_total

            action = self._extract_action(response)
            if action is not None:
                # Return only the canonical action.  This is what results.json and
                # the action parser see; reasoning_content is deliberately ignored.
                return action, elapsed_total

            LOG.warning(
                "[%s] response contained no legal action (attempt %d/%d, finish=%s)",
                self.name,
                attempt + 1,
                attempts,
                self._finish_reason(response),
            )

        # Preserve something useful in the transcript if every forced attempt
        # failed.  ActionParser will mark this as parse_ok=False and use fallback.
        return self._response_text(last_response), elapsed_total

    def _request_params(self, messages: list[dict], token_budget: int) -> dict:
        params = {
            "model": self._model_id,
            "messages": messages,
            "max_completion_tokens": token_budget,
            "temperature": self._temperature,
        }

        if self._reasoning_effort:
            params["reasoning_effort"] = self._reasoning_effort

        if self._force_action:
            params["tools"] = [self._action_tool()]
            params["tool_choice"] = {
                "type": "function",
                "function": {"name": "choose_action"},
            }

        return params

    @staticmethod
    def _action_tool() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "choose_action",
                "description": "Select exactly one legal Crafter action for this turn.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": list(ACTIONS),
                        }
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
            },
        }

    def _create_response(self, params: dict):
        """Call the API and return the full response plus elapsed seconds.

        Unsupported optional parameters are removed and retried.  This matters
        because different Hugging Face providers support different subsets of
        the OpenAI-compatible API.
        """
        import openai

        rate_limit_attempts = 0
        while True:
            try:
                start = time.perf_counter()
                response = self._client.chat.completions.create(**params)
                elapsed = time.perf_counter() - start
                return response, elapsed
            except openai.BadRequestError as exc:
                bad = self._rejected_param(exc)
                if bad and bad in params and bad not in ("model", "messages"):
                    LOG.warning("[%s] endpoint rejected '%s'; retrying without it.", self.name, bad)
                    params.pop(bad)
                    continue
                raise
            except (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
            ) as exc:
                rate_limit_attempts += 1
                if rate_limit_attempts > self._max_retries:
                    raise
                wait = self._retry_wait(exc, rate_limit_attempts)
                LOG.warning(
                    "[%s] rate limited / transient error - waiting %.1fs, retry %d/%d",
                    self.name,
                    wait,
                    rate_limit_attempts,
                    self._max_retries,
                )
                time.sleep(wait)

    # -- response extraction --------------------------------------------------
    def _extract_action(self, response) -> str | None:
        if response is None or not getattr(response, "choices", None):
            return None

        message = response.choices[0].message

        # Preferred path: the model obeyed the forced tool call.
        for tool_call in getattr(message, "tool_calls", None) or []:
            function = getattr(tool_call, "function", None)
            if getattr(function, "name", None) != "choose_action":
                continue
            action = self._action_from_arguments(getattr(function, "arguments", ""))
            if action is not None:
                return action

        # Compatibility path for providers that ignore/reject tool_choice but
        # still print an action in normal content.
        action = self._find_action(getattr(message, "content", None) or "")
        if action is not None:
            return action

        # Last-resort compatibility path.  Some OpenAI-compatible providers put
        # all generated text in reasoning_content.  This is not preferred, but
        # it is better than turning a clearly named action into a noop.
        return self._find_action(self._reasoning_text(message))

    @staticmethod
    def _action_from_arguments(arguments) -> str | None:
        if isinstance(arguments, dict):
            value = arguments.get("action")
            return OpenAIModel._canonical_action(value)

        try:
            parsed = json.loads(arguments or "{}")
        except (TypeError, json.JSONDecodeError):
            return OpenAIModel._find_action(str(arguments or ""))

        if isinstance(parsed, dict):
            return OpenAIModel._canonical_action(parsed.get("action"))
        return None

    @staticmethod
    def _canonical_action(value) -> str | None:
        if not isinstance(value, str):
            return None
        lowered = value.strip().lower()
        return lowered if lowered in ACTIONS else None

    @staticmethod
    def _find_action(text: str) -> str | None:
        hits = _ACTION_RE.findall(text or "")
        return hits[-1].lower() if hits else None

    def _response_text(self, response) -> str:
        if response is None or not getattr(response, "choices", None):
            return ""
        message = response.choices[0].message
        content = (getattr(message, "content", None) or "").strip()
        if content:
            return content
        return self._reasoning_text(message).strip()

    @staticmethod
    def _reasoning_text(message) -> str:
        value = getattr(message, "reasoning_content", None)
        if value:
            return str(value)
        extra = getattr(message, "model_extra", None)
        if isinstance(extra, dict):
            value = extra.get("reasoning_content") or extra.get("reasoning")
            if value:
                return str(value)
        return ""

    @staticmethod
    def _finish_reason(response) -> str | None:
        if response is None or not getattr(response, "choices", None):
            return None
        return getattr(response.choices[0], "finish_reason", None)

    # -- retry helpers --------------------------------------------------------
    def _retry_wait(self, exc, attempt: int) -> float:
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