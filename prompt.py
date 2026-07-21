"""
prompt.py
=========

Builds the system/user prompt shown to the model each turn by filling the
templates in ``config.yaml`` with the current world observation.

Placeholders supported in the user template:
    {objective} {map} {legend} {inventory} {achievements}
    {actions} {position} {facing}

AI-facing: no prints, no inputs.
"""

from __future__ import annotations

from config import PromptCfg
import observation as obs
from actions import ACTIONS


class PromptBuilder:
    """Assembles the per-turn prompt from templates + live observation."""

    def __init__(self, prompt_cfg: PromptCfg, objective_label: str):
        self._cfg = prompt_cfg
        self._objective = objective_label
        self._action_list = "\n".join(f"  - {a}" for a in ACTIONS)

    def build(self, env) -> tuple[str, str]:
        """Return ``(system_prompt, user_prompt)`` for the current state."""
        world, player = env.world, env.player

        legend = obs.build_legend(world, player) if self._cfg.include_legend else "(hidden)"
        inventory = obs.format_inventory(player.inventory) if self._cfg.include_inventory else "(hidden)"
        achievements = obs.format_achievements(player.achievements) if self._cfg.include_achievements else "(hidden)"
        actions = self._action_list if self._cfg.include_action_list else "(hidden)"

        fields = dict(
            objective=self._objective,
            map=obs.render_text_map(world, player),
            legend=legend,
            inventory=inventory,
            achievements=achievements,
            actions=actions,
            position=obs.describe_position(player),
            facing=obs.describe_facing(player),
        )
        # Both templates get the same placeholders, so constant context (the
        # goal, the action list) can live in the system prompt and per-turn state
        # in the user prompt - whichever the config author prefers.
        system = self._safe_format(self._cfg.system, fields)
        user = self._safe_format(self._cfg.user, fields)
        return system, user

    @staticmethod
    def _safe_format(template: str, fields: dict) -> str:
        """Fill {placeholders} but leave any other braces untouched, so a stray
        '{' in a prompt can't crash the run."""
        import string

        class _Safe(dict):
            def __missing__(self, key):
                return "{" + key + "}"

        return string.Formatter().vformat(template, (), _Safe(fields))