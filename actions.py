"""
actions.py
==========

Turns a model's raw text reply into a concrete Crafter action index.

Strategy 'keyword':
  1. If a line contains "ACTION:", search the remainder of that line for a
     valid action name.
  2. Otherwise scan the whole reply and take the LAST valid action name
     mentioned (models tend to reason first, then conclude).
  3. If nothing valid is found, use the configured fallback and flag the parse
     as failed so it shows up in the results.

AI-facing: no prints, no inputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import crafter.constants as C

ACTIONS: tuple[str, ...] = tuple(C.actions)

# Longest names first so e.g. "make_wood_pickaxe" wins over any short overlap.
_ACTION_ALTERNATION = "|".join(
    re.escape(a) for a in sorted(ACTIONS, key=len, reverse=True)
)
_ACTION_RE = re.compile(rf"\b({_ACTION_ALTERNATION})\b", re.IGNORECASE)
_ACTION_LINE_RE = re.compile(r"action\s*[:\-]\s*(.+)", re.IGNORECASE)


def action_index(name: str) -> int:
    """Map an action name to its Crafter action index."""
    return ACTIONS.index(name)


@dataclass
class ParsedAction:
    """Result of parsing one model reply."""

    name: str
    index: int
    ok: bool          # True if a valid action was actually found in the text


class ActionParser:
    """Config-driven parser. Currently supports the 'keyword' strategy."""

    def __init__(self, strategy: str = "keyword", fallback: str = "noop"):
        self._strategy = strategy
        self._fallback = fallback if fallback in ACTIONS else "noop"

    def parse(self, raw_text: str) -> ParsedAction:
        name = self._extract(raw_text or "")
        if name is None:
            return ParsedAction(self._fallback, action_index(self._fallback), ok=False)
        return ParsedAction(name, action_index(name), ok=True)

    # -- internals ------------------------------------------------------------
    def _extract(self, text: str) -> str | None:
        # 1) Prefer an explicit "ACTION:" line.
        for match in _ACTION_LINE_RE.finditer(text):
            found = _ACTION_RE.search(match.group(1))
            if found:
                return found.group(1).lower()

        # 2) Fall back to the last valid action mentioned anywhere.
        all_hits = _ACTION_RE.findall(text)
        if all_hits:
            return all_hits[-1].lower()

        return None
