"""
success.py
==========

The single, swappable definition of "did the agent succeed?".

Change the objective in ``config.yaml`` (e.g. ``collect_wood`` ->
``make_stone_pickaxe``) and this module reads the new target automatically.
Two objective kinds are supported:

* ``achievement`` - success once a Crafter achievement is unlocked.
* ``inventory``   - success once the player holds >= N of an item.
"""

from __future__ import annotations

from config import ObjectiveCfg


class ObjectiveChecker:
    """Evaluates whether the current environment state satisfies the goal."""

    def __init__(self, objective: ObjectiveCfg):
        self._obj = objective

    @property
    def label(self) -> str:
        return self._obj.label

    def is_success(self, env) -> bool:
        """True once the objective is met."""
        if self._obj.type == "inventory":
            return env.player.inventory.get(self._obj.item, 0) >= self._obj.amount
        return env.player.achievements.get(self._obj.target, 0) > 0

    def progress(self, env) -> int:
        """A raw progress number (count) for logging/inspection."""
        if self._obj.type == "inventory":
            return int(env.player.inventory.get(self._obj.item, 0))
        return int(env.player.achievements.get(self._obj.target, 0))
