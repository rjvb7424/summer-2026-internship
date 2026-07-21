"""
models/conversation.py
======================

Short-term conversation memory so a model remembers its own recent turns.

Each turn the runner sends a fresh (system, user) prompt describing the current
state. Without memory the model is amnesiac - it re-derives everything every
turn and will happily walk into the same water tile twice. This keeps a sliding
window of recent (user, assistant) exchanges and prepends them, so the model can
see what it just did and how the state changed as a result.

``max_turns`` controls the window:
    0    -> disabled (stateless: one turn at a time, the old behaviour)
    N>0  -> keep the last N exchanges
    <0   -> keep the whole episode (careful: tokens & cost grow every turn)

Call ``reset()`` at the start of every trial so episodes don't bleed together.
"""

from __future__ import annotations


class ConversationMemory:
    """A windowed history of (user, assistant) turns for one episode."""

    def __init__(self, max_turns: int = 0):
        self.max_turns = int(max_turns)
        self._turns: list[tuple[str, str]] = []

    def reset(self) -> None:
        self._turns.clear()

    def _window(self) -> list[tuple[str, str]]:
        if self.max_turns == 0:
            return []
        if self.max_turns < 0:
            return self._turns
        return self._turns[-self.max_turns:]

    # -- OpenAI / HF / chat-template format -----------------------------------
    def messages(self, system_prompt: str, user_prompt: str) -> list[dict]:
        """[system?, (user, assistant) * window, current user]."""
        msgs: list[dict] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        for user, assistant in self._window():
            msgs.append({"role": "user", "content": user})
            msgs.append({"role": "assistant", "content": assistant})
        msgs.append({"role": "user", "content": user_prompt})
        return msgs

    # -- Gemini format (role "model", parts list, system kept separate) -------
    def gemini_contents(self, user_prompt: str) -> list[dict]:
        contents: list[dict] = []
        for user, assistant in self._window():
            contents.append({"role": "user", "parts": [{"text": user}]})
            contents.append({"role": "model", "parts": [{"text": assistant}]})
        contents.append({"role": "user", "parts": [{"text": user_prompt}]})
        return contents

    def record(self, user_prompt: str, assistant_reply: str) -> None:
        if self.max_turns == 0:
            return
        self._turns.append((user_prompt, assistant_reply or ""))
