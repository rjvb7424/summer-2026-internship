"""
models/openai_api.py
====================

OpenAI-compatible backend used for both OpenAI models and Hugging Face
Inference Providers.

When ``force_action`` is enabled, the backend tries to force a structured
``choose_action`` tool call constrained to Crafter's legal actions. Some hosted
models do not support tools. In that case, the backend automatically retries
without tools and remembers that limitation for later turns.
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
    rf"\b({
        '|'.join(
            re.escape(action)
            for action in sorted(ACTIONS, key=len, reverse=True)
        )
    })\b",
    re.IGNORECASE,
)


class OpenAIModel(LanguageModel):
    """Call an OpenAI-compatible chat-completions endpoint once per turn."""

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

        # None: tool support has not been tested.
        # True: tools worked.
        # False: the model/provider rejected tools.
        #
        # Once a hosted model rejects tools, later turns skip them immediately.
        self._tools_supported: bool | None = None

        # Same idea for temperature: reasoning models (o1/o3/...) reject a custom
        # temperature. Once rejected, stop sending it so we don't waste a failed
        # request every single turn.
        #
        # Reasoning models are detectable up front (o-series names, or an
        # explicit reasoning_effort), so we skip temperature from the very first
        # turn - no failed probe, no warning at all.
        self._temperature_supported: bool | None = None
        if self._is_reasoning_model():
            self._temperature_supported = False

    def _is_reasoning_model(self) -> bool:
        """True for models that reject a custom temperature (OpenAI o-series,
        or anything given a reasoning_effort)."""
        if self._reasoning_effort:
            return True
        return bool(re.match(r"o\d", (self._model_id or ""), re.IGNORECASE))

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
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

        self._client = OpenAI(
            api_key=api_key,
            base_url=self._base_url,
        )

    def unload(self) -> None:
        self._client = None

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, float]:
        base_messages: list[dict] = []

        if system_prompt:
            base_messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        base_messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )

        # Request throttling is deliberately excluded from measured model
        # latency.
        if self._request_delay > 0:
            time.sleep(self._request_delay)

        attempts = 1 + (
            self._action_retries
            if self._force_action
            else 0
        )

        elapsed_total = 0.0
        last_response = None
        self.last_usage = 0  # accumulate tokens across attempts this turn

        for attempt in range(attempts):
            messages = list(base_messages)

            if attempt > 0:
                if self._tools_supported is False:
                    retry_instruction = (
                        "The previous answer contained no valid action. "
                        "Return exactly one action name copied from "
                        "AVAILABLE ACTIONS. Return no other text."
                    )
                else:
                    retry_instruction = (
                        "The previous attempt did not select a legal action. "
                        "Choose exactly one action now using choose_action."
                    )

                messages.append(
                    {
                        "role": "user",
                        "content": retry_instruction,
                    }
                )

            # Reasoning tokens count toward the completion budget. A retry gets
            # progressively more room, up to the configured safety cap.
            token_budget = min(
                self._max_tokens * (2 ** attempt),
                MAX_ACTION_TOKEN_BUDGET,
            )

            params = self._request_params(
                messages=messages,
                token_budget=token_budget,
            )

            response, elapsed = self._create_response(params)

            elapsed_total += elapsed
            last_response = response

            if not self._force_action:
                return self._response_text(response), elapsed_total

            action = self._extract_action(response)

            if action is not None:
                # The experiment receives only the canonical action, not the
                # reasoning or other model output.
                return action, elapsed_total

            LOG.warning(
                "[%s] response contained no legal action "
                "(attempt %d/%d, finish=%s)",
                self.name,
                attempt + 1,
                attempts,
                self._finish_reason(response),
            )

        # If every attempt fails, preserve the raw output. ActionParser will
        # mark the parse as failed and apply the configured fallback.
        return self._response_text(last_response), elapsed_total

    def _request_params(
        self,
        messages: list[dict],
        token_budget: int,
    ) -> dict:
        params = {
            "model": self._model_id,
            "messages": messages,
            "max_completion_tokens": token_budget,
        }

        if self._temperature_supported is not False:
            params["temperature"] = self._temperature

        if self._reasoning_effort:
            params["reasoning_effort"] = self._reasoning_effort

        # Try structured action selection unless this model/provider has already
        # rejected tool calling.
        if self._force_action and self._tools_supported is not False:
            params["tools"] = [self._action_tool()]

            params["tool_choice"] = {
                "type": "function",
                "function": {
                    "name": "choose_action",
                },
            }

        return params

    @staticmethod
    def _action_tool() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "choose_action",
                "description": (
                    "Select exactly one legal Crafter action for this turn."
                ),
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

    # -------------------------------------------------------------------------
    # API request
    # -------------------------------------------------------------------------
    def _create_response(self, params: dict):
        """Return the full API response and measured request time.

        Unsupported optional parameters are removed and retried.

        If a hosted model rejects tool calling, the same request is retried as
        an ordinary text completion. Future turns for that model then skip tools.
        """
        import openai

        rate_limit_attempts = 0

        while True:
            try:
                start = time.perf_counter()

                response = self._client.chat.completions.create(
                    **params
                )

                elapsed = time.perf_counter() - start

                # Accumulate real token usage (prompt + completion, and for
                # reasoning models the hidden reasoning tokens are included in
                # completion_tokens). Guarded so a provider that omits usage
                # never breaks the run.
                usage = getattr(response, "usage", None)
                total = getattr(usage, "total_tokens", None) if usage else None
                if total is not None:
                    self.last_usage = (self.last_usage or 0) + int(total)

                if "tools" in params:
                    self._tools_supported = True

                return response, elapsed

            except openai.BadRequestError as exc:
                # Some endpoints return a 400 rather than a 405 when tools are
                # unsupported.
                if (
                    "tools" in params
                    and self._is_tools_unsupported(exc)
                ):
                    self._disable_tools_and_retry(params)
                    continue

                bad_parameter = self._rejected_param(exc)

                if (
                    bad_parameter
                    and bad_parameter in params
                    and bad_parameter not in ("model", "messages")
                ):
                    LOG.warning(
                        "[%s] endpoint rejected '%s'; retrying without it.",
                        self.name,
                        bad_parameter,
                    )

                    # tools and tool_choice depend on each other.
                    if bad_parameter in ("tools", "tool_choice"):
                        self._disable_tools_and_retry(params)
                    elif bad_parameter == "temperature":
                        # Remember it so future turns don't re-send it.
                        self._temperature_supported = False
                        params.pop("temperature", None)
                    else:
                        params.pop(bad_parameter, None)

                    continue

                raise

            except openai.APIStatusError as exc:
                # Hugging Face may return HTTP 405 for models such as Phi-4 that
                # do not support tool calling.
                if (
                    "tools" in params
                    and self._is_tools_unsupported(exc)
                ):
                    self._disable_tools_and_retry(params)
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

                wait = self._retry_wait(
                    exc,
                    rate_limit_attempts,
                )

                LOG.warning(
                    "[%s] rate limited / transient error - "
                    "waiting %.1fs, retry %d/%d",
                    self.name,
                    wait,
                    rate_limit_attempts,
                    self._max_retries,
                )

                time.sleep(wait)

    def _disable_tools_and_retry(self, params: dict) -> None:
        self._tools_supported = False

        params.pop("tools", None)
        params.pop("tool_choice", None)

        LOG.warning(
            "[%s] tool calling is unsupported; "
            "retrying with plain text output.",
            self.name,
        )

    @staticmethod
    def _is_tools_unsupported(exc) -> bool:
        text_parts = [str(exc)]

        body = getattr(exc, "body", None)

        if isinstance(body, dict):
            error = body.get("error", body)

            if isinstance(error, dict):
                for key in (
                    "message",
                    "type",
                    "param",
                    "code",
                ):
                    text_parts.append(
                        str(error.get(key, ""))
                    )
            else:
                text_parts.append(str(error))

        text = " ".join(text_parts).lower()

        unsupported_phrases = (
            "tool calling is not supported",
            "tool calling not supported",
            "tools are not supported",
            "tools not supported",
            "does not support tool",
            "doesn't support tool",
            "unsupported tool",
        )

        return any(
            phrase in text
            for phrase in unsupported_phrases
        )

    # -------------------------------------------------------------------------
    # Response extraction
    # -------------------------------------------------------------------------
    def _extract_action(self, response) -> str | None:
        if (
            response is None
            or not getattr(response, "choices", None)
        ):
            return None

        message = response.choices[0].message

        # Preferred path: structured choose_action call.
        for tool_call in (
            getattr(message, "tool_calls", None) or []
        ):
            function = getattr(
                tool_call,
                "function",
                None,
            )

            if (
                getattr(function, "name", None)
                != "choose_action"
            ):
                continue

            action = self._action_from_arguments(
                getattr(function, "arguments", "")
            )

            if action is not None:
                return action

        # Compatibility path: normal visible response text.
        action = self._find_action(
            getattr(message, "content", None) or ""
        )

        if action is not None:
            return action

        # Some OpenAI-compatible endpoints put generated text into a nonstandard
        # reasoning_content field.
        return self._find_action(
            self._reasoning_text(message)
        )

    @staticmethod
    def _action_from_arguments(arguments) -> str | None:
        if isinstance(arguments, dict):
            return OpenAIModel._canonical_action(
                arguments.get("action")
            )

        try:
            parsed = json.loads(arguments or "{}")
        except (
            TypeError,
            json.JSONDecodeError,
        ):
            return OpenAIModel._find_action(
                str(arguments or "")
            )

        if isinstance(parsed, dict):
            return OpenAIModel._canonical_action(
                parsed.get("action")
            )

        return None

    @staticmethod
    def _canonical_action(value) -> str | None:
        if not isinstance(value, str):
            return None

        lowered = value.strip().lower()

        if lowered in ACTIONS:
            return lowered

        return None

    @staticmethod
    def _find_action(text: str) -> str | None:
        hits = _ACTION_RE.findall(text or "")

        if not hits:
            return None

        return hits[-1].lower()

    def _response_text(self, response) -> str:
        if (
            response is None
            or not getattr(response, "choices", None)
        ):
            return ""

        message = response.choices[0].message
        content = getattr(message, "content", None)

        if isinstance(content, str) and content.strip():
            return content.strip()

        return self._reasoning_text(message).strip()

    @staticmethod
    def _reasoning_text(message) -> str:
        value = getattr(
            message,
            "reasoning_content",
            None,
        )

        if value:
            return str(value)

        extra = getattr(
            message,
            "model_extra",
            None,
        )

        if isinstance(extra, dict):
            value = (
                extra.get("reasoning_content")
                or extra.get("reasoning")
            )

            if value:
                return str(value)

        return ""

    @staticmethod
    def _finish_reason(response) -> str | None:
        if (
            response is None
            or not getattr(response, "choices", None)
        ):
            return None

        return getattr(
            response.choices[0],
            "finish_reason",
            None,
        )

    # -------------------------------------------------------------------------
    # Retry helpers
    # -------------------------------------------------------------------------
    def _retry_wait(
        self,
        exc,
        attempt: int,
    ) -> float:
        retry_after = self._retry_after(exc)

        if retry_after is not None:
            return retry_after

        backoff = (
            self._retry_base_delay
            * (2 ** (attempt - 1))
        )

        return (
            min(backoff, MAX_BACKOFF)
            + random.uniform(0, 0.5)
        )

    @staticmethod
    def _retry_after(exc) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)

        if headers:
            value = headers.get("retry-after")

            if value:
                try:
                    return float(value)
                except (
                    TypeError,
                    ValueError,
                ):
                    return None

        return None

    @staticmethod
    def _rejected_param(exc) -> str | None:
        body = getattr(exc, "body", None)

        if isinstance(body, dict):
            error = body.get("error", body)

            if isinstance(error, dict):
                return error.get("param")

        return None