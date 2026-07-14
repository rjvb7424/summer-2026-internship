"""All prompt engineering for the Crafter agent in one place.

Owns: the system prompt (static rules), the user prompt (current world state),
world-state formatting, action parsing, and action-result descriptions.
No game loop logic lives here.
"""

import itertools
import re

import crafter

# Local view half-size: the model sees a (2R+1) x (2R+1) grid around the player.
VIEW_RADIUS = 4
FACING_NAMES = {(-1, 0): "west", (1, 0): "east", (0, -1): "north", (0, 1): "south"}
STATUS_KEYS = ("health", "food", "drink", "energy")

SYSTEM_PROMPT = """
You are in a 2D survival simulation.

Your objective is to survive for as long as possible,
while completing as many achievements as you can!

You will be given a description of the current state of the world, your inventory,
and your recent actions. You must choose one action to take from the list of valid actions.
You will receive feedback on the result of your action,
and you should use that feedback to inform your next choice.

The list of valid actions is: {actions_list}.
The list of achievements you should pursue is: {achievements_list}.

How to interpret the world state:
- You are at the center of the world view, shown as YOU.
- The top row of the world view is north, the bottom row is south,
  the left column is west, and the right column is east.
- The world view shows the tiles around you, including resources, objects, and terrain.
- "do" interacts with the tile you are facing (chop tree, mine stone/coal/iron,
  attack, drink water, eat cow). Moving into a blocking tile only turns you to face it.
- Crafting needs a table nearby (place_table costs wood); iron tools also need a
  furnace; place_plant grows food; sleep restores energy at night.

Some important rules to follow when choosing your next action are:
1. Prioritize actions that will ensure your survival, such as gathering food or water when they are low.
2. Choose actions that will help you complete achievements, while also ensuring your survival.
3. Do not repeat an action that was BLOCKED or had NO EFFECT without changing something first.
4. Try to plan ahead and think about the consequences of your actions.

Reply with exactly one action name from the list and nothing else.
""".strip().format(
    actions_list=", ".join(crafter.constants.actions),
    achievements_list=", ".join(crafter.constants.achievements),
)

USER_PROMPT = """MAP ({size}x{size}, you are YOU at the center, top row is north):
{grid}

You are facing {facing}. The tile you are facing is: {faced_tile}.
Status: {status}
Inventory: {inventory}
Achievements unlocked ({unlocked_count}/{total_count}): {unlocked}
Still locked: {locked}

Recent actions and their observed results:
{history}

Choose the single action that best advances a locked achievement while keeping \
you alive. Reply with exactly one action name."""


# ============================================================
# Semantic map
# ============================================================
def build_id_map(env):
    """Map semantic-view ids to readable names (materials and objects)."""
    id_to_name = {0: "empty"}
    pairs = itertools.chain(
        env._world._mat_ids.items(), env._sem_view._obj_ids.items()
    )
    for key, index in pairs:
        if key is None:
            continue
        name = key if isinstance(key, str) else key.__name__.lower()
        id_to_name[index] = str(name)
    return id_to_name


def tile_name(semantic, x, y, id_to_name):
    width, height = semantic.shape
    if 0 <= x < width and 0 <= y < height:
        return id_to_name[int(semantic[x, y])]
    return "void"


def local_grid(semantic, player_position, id_to_name):
    """Readable (2R+1)^2 grid around the player, top row = north."""
    x0, y0 = int(player_position[0]), int(player_position[1])
    rows = []
    for y in range(y0 - VIEW_RADIUS, y0 + VIEW_RADIUS + 1):
        cells = [
            "YOU" if (x == x0 and y == y0) else tile_name(semantic, x, y, id_to_name)
            for x in range(x0 - VIEW_RADIUS, x0 + VIEW_RADIUS + 1)
        ]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


# ============================================================
# State snapshots and action feedback
# ============================================================
def snapshot(env, info):
    """Capture the observable state used to describe an action's effect."""
    return {
        "position": tuple(int(v) for v in info["player_pos"]),
        "facing": tuple(env._player.facing),
        "inventory": dict(info["inventory"]),
        "achievements": dict(info["achievements"]),
    }


def describe_result(action, before, after):
    """Describe the observable effect of the previous action."""
    new_achievements = [
        name for name, count in after["achievements"].items()
        if count > before["achievements"].get(name, 0)
    ]
    if new_achievements:
        return "SUCCESS; unlocked " + ", ".join(new_achievements)

    inventory_changes = [
        f"{name} {before['inventory'].get(name, 0)}->{amount}"
        for name, amount in after["inventory"].items()
        if amount != before["inventory"].get(name, 0)
    ]
    if inventory_changes:
        return "SUCCESS; inventory changed: " + ", ".join(inventory_changes)

    if action.startswith("move_"):
        if after["position"] != before["position"]:
            return f"SUCCESS; moved from {before['position']} to {after['position']}"
        if after["facing"] != before["facing"]:
            return "BLOCKED; only turned to face that direction"
        return "BLOCKED; position and facing did not change"

    return "NO EFFECT; nothing changed"


# ============================================================
# Prompt building and action parsing
# ============================================================
def build_user_prompt(env, info, history, id_to_name):
    """Build the user prompt (current world state) from the environment."""
    inventory = info["inventory"]
    semantic = info["semantic"]
    x0, y0 = int(info["player_pos"][0]), int(info["player_pos"][1])
    facing = tuple(env._player.facing)

    status = ", ".join(f"{key}: {inventory[key]}" for key in STATUS_KEYS)
    items = ", ".join(
        f"{key}: {count}" for key, count in inventory.items()
        if count > 0 and key not in STATUS_KEYS
    ) or "nothing"
    unlocked = [name for name, count in info["achievements"].items() if count > 0]
    locked = [name for name, count in info["achievements"].items() if count == 0]
    history_text = "\n".join(
        f"- {action} -> {result}" for action, result in history
    ) or "- none yet"

    return USER_PROMPT.format(
        size=2 * VIEW_RADIUS + 1,
        grid=local_grid(semantic, info["player_pos"], id_to_name),
        facing=FACING_NAMES[facing],
        faced_tile=tile_name(semantic, x0 + facing[0], y0 + facing[1], id_to_name),
        status=status,
        inventory=items,
        unlocked_count=len(unlocked),
        total_count=len(info["achievements"]),
        unlocked=", ".join(unlocked) or "none",
        locked=", ".join(locked) or "none",
        history=history_text,
    )


def parse_action(text, action_names):
    """Return the index of the chosen action, or None if unparseable."""
    if not text:
        return None
    text = text.strip().lower()

    # 1) The whole response is exactly one action name.
    if text in action_names:
        return action_names.index(text)

    # 2) Explicit "action: X" style statements.
    for pattern in (
        r'"action"\s*:\s*"([a-z_]+)"',
        r"action\s*[:=]\s*([a-z_]+)",
        r"final action\s*[:=]?\s*([a-z_]+)",
    ):
        match = re.search(pattern, text)
        if match and match.group(1) in action_names:
            return action_names.index(match.group(1))

    # 3) Otherwise, the last action name mentioned anywhere.
    matches = [
        (match.start(), action)
        for action in action_names
        for match in re.finditer(rf"\b{re.escape(action)}\b", text)
    ]
    return action_names.index(max(matches)[1]) if matches else None