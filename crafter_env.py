import numpy as np

MATERIAL_SYMBOLS = {
    None: "?",
    "water": "~",
    "grass": ".",
    "stone": "#",
    "path": ":",
    "sand": ",",
    "tree": "T",
    "lava": "L",
    "coal": "c",
    "iron": "i",
    "diamond": "d",
    "table": "b",
    "furnace": "f",
}
ENTITY_SYMBOLS = {
    "Player": "@",
    "Cow": "C",
    "Zombie": "Z",
    "Skeleton": "S",
    "Arrow": "A",
    "Plant": "P",
}
FACING_NAMES = {
    (-1, 0): "west/left",
    (1, 0): "east/right",
    (0, -1): "north/up",
    (0, 1): "south/down",
}
DIRECTION_VECTORS = {
    "move_left": (-1, 0),
    "move_right": (1, 0),
    "move_up": (0, -1),
    "move_down": (0, 1),
}
SCENE_LEGEND = (
    "Legend: @ player, . grass, ~ water, # stone, : path, , sand, "
    "T tree, L lava, c coal, i iron, d diamond, b table, f furnace, "
    "C cow, Z zombie, S skeleton, A arrow, P plant, ? unknown."
)

def player_position(env):
    """Return the player's current world position."""
    return tuple(int(value) for value in env._player.pos)

def facing_name(env):
    """Return the human-readable direction the player is facing."""
    return FACING_NAMES.get(tuple(env._player.facing), "unknown")

def format_inventory(env):
    """Format current inventory and survival values."""
    return ", ".join(
        f"{name}={amount}"
        for name, amount in env._player.inventory.items()
    )

def format_adjacent_tiles(env):
    """Describe the four tiles directly adjacent to the player."""
    world = env._world
    player = env._player
    rows = []

    for action, direction in DIRECTION_VECTORS.items():
        target = player.pos + np.array(direction)
        material, obj = world[target]
        direction_name = action.removeprefix("move_")

        if obj is not None:
            contents = type(obj).__name__
            walkability = "blocked"
        else:
            contents = material
            walkability = (
                "walkable"
                if material in player.walkable
                else "blocked"
            )

        facing_marker = (
            " [FACING]"
            if tuple(direction) == tuple(player.facing)
            else ""
        )
        rows.append(
            f"- {direction_name}: {contents}, "
            f"{walkability}{facing_marker}"
        )

    return "\n".join(rows)

def format_visible_scene(env):
    """Return a string representation of the visible scene around the player."""
    world = env._world
    player = env._player
    grid = np.array(env._local_view._grid)
    offset = grid // 2
    rows = []

    for local_y in range(grid[1]):
        symbols = []
        for local_x in range(grid[0]):
            position = (
                player.pos
                + np.array((local_x, local_y))
                - offset
            )
            material, obj = world[position]

            if obj is not None:
                symbol = ENTITY_SYMBOLS.get(type(obj).__name__, "O")
            else:
                symbol = MATERIAL_SYMBOLS.get(material, "?")
            symbols.append(symbol)
        rows.append(" ".join(symbols))

    return "\n".join(rows)
