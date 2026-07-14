from __future__ import annotations

from typing import Any

import crafter
from crafter import objects

import config


DIRECTION_VECTORS = {
    "left": (-1, 0),
    "right": (1, 0),
    "up": (0, -1),
    "down": (0, 1),
}

OBJECT_NAMES = {
    objects.Player: "player",
    objects.Cow: "cow",
    objects.Zombie: "zombie",
    objects.Skeleton: "skeleton",
    objects.Arrow: "arrow",
    objects.Plant: "plant",
    objects.Fence: "fence",
}

MAP_SYMBOLS = {
    "grass": ".",
    "path": ".",
    "tree": "T",
    "stone": "S",
    "water": "W",
    "lava": "L",
    "coal": "C",
    "iron": "I",
    "diamond": "D",
    "sand": "s",
    "table": "B",
    "furnace": "F",
    "player": "P",
    "cow": "c",
    "zombie": "z",
    "skeleton": "k",
    "arrow": "a",
    "plant": "p",
    "fence": "f",
}


class FixedCrafterWorld:
    """Creates and exposes one deterministic, fixed Crafter world."""

    def __init__(self, seed: int):
        self.seed = seed
        self.env = self._build()

    @property
    def world(self):
        return self.env._world

    @property
    def player(self):
        return self.env._player

    def _build(self):
        env = crafter.Env(
            area=(config.WORLD_SIZE, config.WORLD_SIZE),
            view=(config.VIEW_SIZE, config.VIEW_SIZE),
            size=(config.WINDOW_SIZE, config.WINDOW_SIZE),
            length=None,
            seed=self.seed,
        )
        env.reset()

        if not config.NATURAL_MOBS:
            env._balance_chunk = lambda *_args, **_kwargs: None

        if not config.RANDOM_WORLD:
            self._clear_generated_world(env)

        self.env = env
        self._place_player(config.PLAYER_POSITION, config.PLAYER_FACING)
        self._place_tiles()
        self._place_rectangles()
        self._place_mobs()

        return env

    def _check_position(self, position: tuple[int, int]) -> None:
        x, y = position
        if not (
            0 <= x < config.WORLD_SIZE
            and 0 <= y < config.WORLD_SIZE
        ):
            raise ValueError(
                f"Position {position} is outside the "
                f"{config.WORLD_SIZE}x{config.WORLD_SIZE} world."
            )

    def _clear_generated_world(self, env) -> None:
        world = env._world
        player = env._player

        for obj in list(world.objects):
            if obj is not player:
                world.remove(obj)

        if config.BACKGROUND not in world._mat_ids:
            raise ValueError(
                f"Unknown BACKGROUND material: {config.BACKGROUND!r}"
            )

        world._mat_map[:] = world._mat_ids[config.BACKGROUND]

    def _place_player(
        self,
        position: tuple[int, int],
        facing: str,
    ) -> None:
        self._check_position(position)

        if facing not in DIRECTION_VECTORS:
            raise ValueError(
                f"Unknown PLAYER_FACING {facing!r}. "
                f"Choose from {sorted(DIRECTION_VECTORS)}."
            )

        if tuple(self.player.pos) != tuple(position):
            _, existing = self.world[position]
            if existing is not None and existing is not self.player:
                self.world.remove(existing)
            self.world.move(self.player, position)

        self.player.facing = DIRECTION_VECTORS[facing]

    def _put_tile(
        self,
        position: tuple[int, int],
        material: str,
    ) -> None:
        self._check_position(position)

        if material not in self.world._mat_ids:
            available = sorted(
                name for name in self.world._mat_ids if name
            )
            raise ValueError(
                f"Unknown material {material!r}. "
                f"Available: {available}"
            )

        self.world[position] = material

    def _place_tiles(self) -> None:
        for position, material in config.TILES.items():
            self._put_tile(position, material)

    def _place_rectangles(self) -> None:
        for material, x, y, width, height in config.RECTANGLES:
            for tile_x in range(x, x + width):
                for tile_y in range(y, y + height):
                    self._put_tile((tile_x, tile_y), material)

    def _place_mobs(self) -> None:
        factories = {
            "cow": lambda pos: objects.Cow(self.world, pos),
            "zombie": lambda pos: objects.Zombie(
                self.world, pos, self.player
            ),
            "skeleton": lambda pos: objects.Skeleton(
                self.world, pos, self.player
            ),
        }

        for mob_name, x, y in config.MOBS:
            position = (x, y)
            self._check_position(position)

            if mob_name not in factories:
                raise ValueError(
                    f"Unknown mob {mob_name!r}. "
                    f"Available: {sorted(factories)}"
                )

            _, existing = self.world[position]
            if existing is not None:
                raise ValueError(
                    f"Cannot place {mob_name} at {position}; "
                    "the cell already contains an object."
                )

            self.world.add(factories[mob_name](position))

    def execute(self, action_name: str):
        """Execute one Crafter action. Turn counting remains in main.py."""
        if action_name not in self.env.action_names:
            raise ValueError(
                f"Unknown Crafter action {action_name!r}."
            )

        action_index = self.env.action_names.index(action_name)
        return self.env.step(action_index)

    def has_achievement(self, name: str) -> bool:
        return self.player.achievements.get(name, 0) > 0

    def state(self) -> dict[str, Any]:
        return {
            "position": tuple(int(x) for x in self.player.pos),
            "facing": self.facing_name(),
            "inventory": {
                name: int(amount)
                for name, amount in self.player.inventory.items()
                if amount
            },
            "achievements": {
                name: int(amount)
                for name, amount in self.player.achievements.items()
                if amount
            },
        }

    def facing_name(self) -> str:
        facing = tuple(int(x) for x in self.player.facing)
        for name, vector in DIRECTION_VECTORS.items():
            if tuple(vector) == facing:
                return name
        return str(facing)

    def cell_name(self, position: tuple[int, int]) -> str:
        material, obj = self.world[position]

        if obj is not None:
            for object_type, name in OBJECT_NAMES.items():
                if isinstance(obj, object_type):
                    return name
            return type(obj).__name__.lower()

        return material or "outside"

    def front_cell(self) -> tuple[tuple[int, int], str]:
        px, py = (int(x) for x in self.player.pos)
        dx, dy = (int(x) for x in self.player.facing)
        position = (px + dx, py + dy)
        return position, self.cell_name(position)

    def text_map(self) -> str:
        rows = []

        # y is printed vertically and x horizontally.
        header = "    " + " ".join(str(x) for x in range(config.WORLD_SIZE))
        rows.append(header)

        for y in range(config.WORLD_SIZE):
            symbols = []
            for x in range(config.WORLD_SIZE):
                name = self.cell_name((x, y))
                symbols.append(MAP_SYMBOLS.get(name, "?"))
            rows.append(f"{y:>2}: " + " ".join(symbols))

        return "\n".join(rows)

    def observation_text(
        self,
        turn: int,
        history: list[dict[str, Any]],
    ) -> str:
        state = self.state()
        front_position, front_name = self.front_cell()

        recent_history = history[-config.HISTORY_LENGTH:]
        history_text = (
            "\n".join(
                f"- Turn {item['turn']}: chose {item['action']}; "
                f"wood after action={item['wood']}"
                for item in recent_history
            )
            if recent_history
            else "- No previous actions."
        )

        map_text = self.text_map() if config.SHOW_FULL_MAP else "(hidden)"

        return f"""Goal: {config.GOAL_DESCRIPTION}
Current turn: {turn}
Player position: {state['position']}
Player facing: {state['facing']}
Cell directly in front: {front_position} contains {front_name}
Inventory: {state['inventory']}
Achievements: {state['achievements']}

Map coordinates use (x, y).
Legend: P=player, T=tree, S=stone, W=water, .=grass

{map_text}

Recent actions:
{history_text}

Important mechanics:
- A movement action changes the facing direction and attempts to move one tile.
- Trees block movement.
- To collect wood, stand next to a tree, face it, then use "do".
- Return exactly one action from:
{", ".join(config.VALID_ACTIONS)}
"""

    def render(self):
        return self.env.render(
            (config.WINDOW_SIZE, config.WINDOW_SIZE)
        )
