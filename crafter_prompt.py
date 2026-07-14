"""Build prompts for the Crafter agent and parse its responses."""

import re

import crafter

from crafter_env import (
    SCENE_LEGEND,
    facing_name,
    format_adjacent_tiles,
    format_inventory,
    format_visible_scene,
    player_position,
)

ACTION_HISTORY_LIMIT = 12

ACTION_NAMES = crafter.constants.actions
ACHIEVEMENTS = crafter.constants.achievements
NUM_ACHIEVEMENTS = len(ACHIEVEMENTS)

_SYSTEM_PROMPT_TEMPLATE = """
You are in a 2D survival simulation.

PRIMARY OBJECTIVE:
To survive as long as possible while completing as many achievements as you can.

You must choose exactly one action per turn. Return only the action name.

VALID ACTIONS:
The list of valid actions is: {actions_list}

WORLD AND CONTROL RULES
- The player is @ at the centre of the visible grid.
- The top of the grid is north/up.
- move_left, move_right, move_up, and move_down try to move one tile and also
  change the direction the player is facing.
- Only grass, path, and sand are normally walkable.
- Water, trees, stone, coal, iron, diamond, tables, furnaces, creatures, and
  plants block movement.
- Use do to interact with the tile directly in front of the player.
- If movement is blocked by a useful resource or creature, the movement still
  faces that target; use do next.
- Do not repeatedly alternate between opposite movements.
- Do not repeat an action that had NO EFFECT unless the state has changed.
- Explore deliberately. Prefer directions that have not just been tried.

RECOMMENDED PROGRESSION
- First collect wood.
- Obtain at least 2 wood and place a table.
- Collect more wood and make a wood pickaxe and preferably a wood sword.
- Use the wood pickaxe to collect stone and coal.
- Place a furnace after collecting 4 stone.
- Make stone tools and collect iron.
- With a table and furnace nearby, make iron tools.
- Use the iron pickaxe to collect diamond.
- Pursue independent achievements whenever opportunities appear: drink water,
  collect a sapling, place a plant, place stone, fight creatures safely, eat a
  cow or mature plant, and sleep to wake up.

DECISION POLICY
1. Protect survival when health, food, drink, or energy is dangerously low.
2. Prefer an action that directly advances an incomplete achievement.
3. Use do when facing a useful resource or target.
4. If the previous movement was blocked, do not repeat it. Use do if the object
   ahead is useful; otherwise choose another direction.
5. If recent actions show left-right-left-right or up-down-up-down oscillation,
   break the loop by choosing a different action.
6. Never choose a crafting or placement action without its required resources
   and nearby structures.
7. Output exactly one valid action name and no explanation.
""".strip()

CRAFTER_SYSTEM_PROMPT = _SYSTEM_PROMPT_TEMPLATE.format(
    actions_list=", ".join(ACTION_NAMES),
)

def build_messages(
    env,
    step,
    max_steps,
    action_history,
    last_action,
    last_action_result,
    last_reward,
):
    """Build the system and user messages for the Crafter agent."""
    achievements = env._player.achievements
    completed = [name for name, count in achievements.items() if count > 0]
    remaining = [name for name, count in achievements.items() if count == 0]

    recent_actions = (
        "\n".join(
            f"- {item['action']}: {item['result']}"
            for item in action_history
        )
        if action_history
        else "- none"
    )

    user_feedback = "\n".join([
        "CURRENT STEP FEEDBACK",
        f"Step: {step}/{max_steps}",
        f"World position: {player_position(env)}",
        f"Facing: {facing_name(env)}",
        f"Last action: {last_action or 'none'}",
        f"Last action result: {last_action_result}",
        f"Last reward: {last_reward:.2f}",
        f"Inventory: {format_inventory(env)}",
        f"Completed achievements ({len(completed)}/{NUM_ACHIEVEMENTS}): "
        f"{', '.join(completed) if completed else 'none'}",
        f"Remaining achievements: {', '.join(remaining)}",
        "",
        f"Loop status: {loop_warning(action_history)}",
        "",
        "Adjacent tiles:",
        format_adjacent_tiles(env),
        "",
        "Recent action outcomes:",
        recent_actions,
        "",
        "Visible local scene:",
        format_visible_scene(env),
        SCENE_LEGEND,
        "",
        "Choose the single action that best advances an incomplete "
        "achievement while avoiding loops and preserving survival.",
        "Return exactly one valid action name.",
    ])

    return [
        {"role": "system", "content": CRAFTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_feedback},
    ]


def loop_warning(action_history):
    """Describe repeated or oscillating action patterns."""
    actions = [item["action"] for item in action_history]

    if len(actions) >= 4:
        last_four = actions[-4:]
        oscillations = {
            ("move_left", "move_right", "move_left", "move_right"),
            ("move_right", "move_left", "move_right", "move_left"),
            ("move_up", "move_down", "move_up", "move_down"),
            ("move_down", "move_up", "move_down", "move_up"),
        }
        if tuple(last_four) in oscillations:
            return (
                "OSCILLATION DETECTED. Do not choose either of the last "
                "two opposite movements. Select do or a perpendicular "
                "direction."
            )

    if len(actions) >= 3 and len(set(actions[-3:])) == 1:
        return (
            f"REPETITION DETECTED: {actions[-1]} was selected three "
            "times. Choose a different action."
        )

    return "No movement loop detected."


# ---------------------------------------------------------------------------
# Response parsing and feedback text
# ---------------------------------------------------------------------------

def extract_action(response_text):
    """Extract one Crafter action name from the model response."""
    if not response_text:
        return None

    text = response_text.strip().lower()

    if text in ACTION_NAMES:
        return text

    explicit_patterns = [
        r'"action"\s*:\s*"([a-z_]+)"',
        r"action\s*[:=]\s*([a-z_]+)",
        r"final action\s*[:=]?\s*([a-z_]+)",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, text)
        if match and match.group(1) in ACTION_NAMES:
            return match.group(1)

    matches = []
    for action in ACTION_NAMES:
        for match in re.finditer(rf"\b{re.escape(action)}\b", text):
            matches.append((match.start(), action))

    return max(matches)[1] if matches else None


def describe_action_result(
    action,
    position_before,
    position_after,
    facing_before,
    facing_after,
    inventory_before,
    inventory_after,
    achievements_before,
    achievements_after,
):
    """Describe the observable effect of the previous action."""
    new_achievements = [
        name
        for name, count in achievements_after.items()
        if count > achievements_before.get(name, 0)
    ]
    inventory_changes = [
        f"{name} {inventory_before.get(name, 0)}->{amount}"
        for name, amount in inventory_after.items()
        if amount != inventory_before.get(name, 0)
    ]

    if new_achievements:
        return "SUCCESS; unlocked " + ", ".join(new_achievements)

    if inventory_changes:
        return "SUCCESS; inventory changed: " + ", ".join(inventory_changes)

    if action.startswith("move_"):
        if position_after != position_before:
            return (
                f"SUCCESS; moved from {position_before} "
                f"to {position_after}"
            )
        if facing_after != facing_before:
            return (
                "BLOCKED; position did not change, but facing changed. "
                "Do not repeat this movement."
            )
        return "BLOCKED; position and facing did not change."

    return (
        "NO EFFECT; the action did not change position, inventory, "
        "or achievements. Do not repeat it without a state change."
    )