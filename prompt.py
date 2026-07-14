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

        user = self._cfg.user.format(
            objective=self._objective,
            map=obs.render_text_map(world, player),
            legend=legend,
            inventory=inventory,
            achievements=achievements,
            actions=actions,
            position=obs.describe_position(player),
            facing=obs.describe_facing(player),
        )
        return self._cfg.system, user
