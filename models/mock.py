"""
models/mock.py
==============

Zero-download agents, useful for two things:

* verifying the whole pipeline (run/log/plot/view) without loading any weights,
* providing a non-LLM reference point.

Policies:
  random     - pick a uniformly random action every turn.
  fixed      - always emit one configured action.
  heuristic  - read the ASCII map from the prompt, walk to the nearest goal
               tile, face it, and 'do'. A strong baseline for tool-free
               "collect_X" / "eat_cow" / "collect_drink" style goals.

The heuristic reads only the prompt text (like a real model would), so it keeps
working even if you tweak the map symbols in observation.py.
"""

from __future__ import annotations

import re
import time

import numpy as np

from models.base import LanguageModel
from observation import MATERIAL_SYMBOLS, OBJECT_SYMBOLS

# Objective target -> the map symbol the heuristic should walk toward.
GOAL_SYMBOL_BY_TARGET: dict[str, str] = {
    "collect_wood": "T",
    "collect_drink": "~",
    "collect_stone": "#",
    "collect_coal": "c",
    "collect_iron": "i",
    "collect_diamond": "*",
    "eat_cow": "C",
}

_WALKABLE_SYMBOLS = {".", "-", ":"}
_KNOWN_SYMBOLS = set(MATERIAL_SYMBOLS.values()) | set(OBJECT_SYMBOLS.values()) | {" "}
_PLAYER = OBJECT_SYMBOLS["Player"]
_FACING_RE = re.compile(r"facing\s*[:=]?\s*(left|right|up|down)", re.IGNORECASE)

# direction name -> (dx, dy) in map coordinates (x right, y down)
_DIR_VEC = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}
_VEC_DIR = {v: k for k, v in _DIR_VEC.items()}


class MockModel(LanguageModel):
    """Rule-based agent selectable via the ``policy`` option."""

    def __init__(self, name: str, policy: str = "heuristic",
                 fixed_action: str = "noop", goal_symbol: str | None = None):
        super().__init__(name)
        self._policy = policy
        self._fixed_action = fixed_action
        self._goal_symbol = goal_symbol
        self._rng = np.random.RandomState(0)

    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, float]:
        start = time.perf_counter()
        if self._policy == "fixed":
            action = self._fixed_action
        elif self._policy == "random":
            action = self._random_action()
        else:
            action = self._heuristic_action(user_prompt)
        elapsed = time.perf_counter() - start
        return f"ACTION: {action}", elapsed

    # -- policies -------------------------------------------------------------
    def _random_action(self) -> str:
        from actions import ACTIONS
        return ACTIONS[self._rng.randint(0, len(ACTIONS))]

    def _heuristic_action(self, user_prompt: str) -> str:
        if not self._goal_symbol:
            return self._random_action()

        grid = self._parse_grid(user_prompt)
        if grid is None:
            return self._random_action()

        player = self._find(grid, _PLAYER)
        target = self._find_nearest(grid, self._goal_symbol, player)
        if player is None or target is None:
            return self._random_action()

        px, py = player
        tx, ty = target
        needed = (int(np.sign(tx - px)), int(np.sign(ty - py)))

        # Adjacent to the goal tile: face it, then interact.
        if abs(tx - px) + abs(ty - py) == 1:
            facing = self._parse_facing(user_prompt)
            if facing == needed:
                return "do"
            return "move_" + _VEC_DIR[needed]

        # Otherwise step toward it along the longer axis, preferring walkable.
        return self._step_toward(grid, player, (tx, ty))

    # -- map parsing ----------------------------------------------------------
    def _parse_grid(self, text: str) -> list[str] | None:
        """Extract the largest block of equal-length map rows from the prompt."""
        best: list[str] = []
        current: list[str] = []
        for line in text.splitlines():
            stripped = line.rstrip("\n")
            is_row = len(stripped) >= 5 and set(stripped) <= _KNOWN_SYMBOLS
            if is_row and (not current or len(stripped) == len(current[0])):
                current.append(stripped)
            else:
                if len(current) > len(best):
                    best = current
                current = [stripped] if is_row else []
        if len(current) > len(best):
            best = current
        return best or None

    @staticmethod
    def _find(grid: list[str], symbol: str) -> tuple[int, int] | None:
        for y, row in enumerate(grid):
            x = row.find(symbol)
            if x != -1:
                return (x, y)
        return None

    @staticmethod
    def _find_nearest(grid: list[str], symbol: str, origin) -> tuple[int, int] | None:
        if origin is None:
            return None
        ox, oy = origin
        best, best_d = None, 1e9
        for y, row in enumerate(grid):
            for x, ch in enumerate(row):
                if ch == symbol:
                    d = abs(x - ox) + abs(y - oy)
                    if d < best_d:
                        best, best_d = (x, y), d
        return best

    def _step_toward(self, grid, player, target) -> str:
        px, py = player
        tx, ty = target
        dx, dy = tx - px, ty - py
        # Try the longer axis first, then the other; skip blocked tiles.
        order = [(int(np.sign(dx)), 0), (0, int(np.sign(dy)))]
        if abs(dy) > abs(dx):
            order.reverse()
        for vec in order:
            if vec == (0, 0):
                continue
            nx, ny = px + vec[0], py + vec[1]
            if self._walkable(grid, nx, ny):
                return "move_" + _VEC_DIR[vec]
        # Nothing walkable toward the goal; nudge in any open direction.
        for vec in _VEC_DIR:
            nx, ny = px + vec[0], py + vec[1]
            if self._walkable(grid, nx, ny):
                return "move_" + _VEC_DIR[vec]
        return "noop"

    @staticmethod
    def _walkable(grid, x, y) -> bool:
        if y < 0 or y >= len(grid) or x < 0 or x >= len(grid[y]):
            return False
        return grid[y][x] in _WALKABLE_SYMBOLS

    @staticmethod
    def _parse_facing(text: str):
        m = _FACING_RE.search(text)
        return _DIR_VEC.get(m.group(1).lower()) if m else None
