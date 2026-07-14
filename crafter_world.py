from __future__ import annotations

import crafter
from crafter import objects

import config


class CrafterWorld:
    """Builds and exposes one configured Crafter environment."""

    def __init__(self):
        self.env = None
        self.reset()

    @property
    def world(self):
        return self.env._world

    @property
    def player(self):
        return self.env._player

    def reset(self):
        self.env = crafter.Env(
            area=(config.WORLD_SIZE, config.WORLD_SIZE),
            view=(config.VIEW_SIZE, config.VIEW_SIZE),
            size=(config.WINDOW_SIZE, config.WINDOW_SIZE),
            length=None,
            seed=config.SEED,
        )
        self.env.reset()

        if not config.NATURAL_MOBS:
            self.env._balance_chunk = lambda *_args, **_kwargs: None

        if not config.RANDOM_WORLD:
            self._clear_generated_world()

        self._place_player(config.PLAYER)
        self._place_tiles()
        self._place_rectangles()
        self._place_mobs()

    def _check_position(self, position):
        x, y = position

        if not (
            0 <= x < config.WORLD_SIZE
            and 0 <= y < config.WORLD_SIZE
        ):
            raise ValueError(
                f"Position {position} is outside a "
                f"{config.WORLD_SIZE} x {config.WORLD_SIZE} world."
            )

    def _clear_generated_world(self):
        for obj in list(self.world.objects):
            if obj is not self.player:
                self.world.remove(obj)

        if config.BACKGROUND not in self.world._mat_ids:
            raise ValueError(
                f"Unknown background tile: {config.BACKGROUND!r}"
            )

        self.world._mat_map[:] = self.world._mat_ids[
            config.BACKGROUND
        ]

    def _place_player(self, position):
        self._check_position(position)

        # Do nothing when the player is already in the requested position.
        if tuple(self.player.pos) == tuple(position):
            return

        existing_object = self.world[position][1]

        if existing_object is not None and existing_object is not self.player:
            self.world.remove(existing_object)

        self.world.move(self.player, position)

    def _put_tile(self, position, tile):
        self._check_position(position)

        if tile not in self.world._mat_ids:
            available = sorted(
                name for name in self.world._mat_ids if name
            )
            raise ValueError(
                f"Unknown tile {tile!r}. "
                f"Available tiles: {available}"
            )

        self.world[position] = tile

    def _place_tiles(self):
        for position, tile in config.TILES.items():
            self._put_tile(position, tile)

    def _place_rectangles(self):
        for tile, x, y, width, height in config.RECTANGLES:
            if width <= 0 or height <= 0:
                raise ValueError(
                    "Rectangle width and height must be positive."
                )

            for tile_x in range(x, x + width):
                for tile_y in range(y, y + height):
                    self._put_tile((tile_x, tile_y), tile)

    def _place_mobs(self):
        factories = {
            "cow": lambda position: objects.Cow(
                self.world,
                position,
            ),
            "zombie": lambda position: objects.Zombie(
                self.world,
                position,
                self.player,
            ),
            "skeleton": lambda position: objects.Skeleton(
                self.world,
                position,
                self.player,
            ),
        }

        for mob_name, x, y in config.MOBS:
            position = (x, y)
            self._check_position(position)

            if mob_name not in factories:
                raise ValueError(
                    f"Unknown mob {mob_name!r}. "
                    f"Available mobs: {sorted(factories)}"
                )

            _, existing_object = self.world[position]
            if existing_object is not None:
                raise ValueError(
                    f"Cannot place {mob_name} at {position}: "
                    "the position already contains an object."
                )

            self.world.add(factories[mob_name](position))

    def render(self):
        return self.env.render(
            (config.WINDOW_SIZE, config.WINDOW_SIZE)
        )

    def get_state(self):
        counts = {
            "cows": 0,
            "zombies": 0,
            "skeletons": 0,
        }

        for obj in self.world.objects:
            if isinstance(obj, objects.Cow):
                counts["cows"] += 1
            elif isinstance(obj, objects.Zombie):
                counts["zombies"] += 1
            elif isinstance(obj, objects.Skeleton):
                counts["skeletons"] += 1

        return {
            "player_position": tuple(self.player.pos),
            "player_health": self.player.inventory["health"],
            **counts,
        }
