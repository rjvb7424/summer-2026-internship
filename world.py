"""
world.py
========

A controllable Crafter environment.

``CustomCrafterEnv`` subclasses the real ``crafter.Env`` so it keeps Crafter's
genuine engine, objects and - crucially - its built-in achievement tracking,
while replacing the procedural world generator with a deterministic,
config-driven ``WorldBuilder``. It also freezes daylight and disables the
dynamic monster/animal balancing so an experiment world stays exactly as you
placed it.
"""

from __future__ import annotations

import logging

import numpy as np
import crafter
from crafter import objects

from config import WorldCfg

LOG = logging.getLogger("crafter_experiment.world")

# Entity name -> constructor. Zombies/skeletons need the player reference.
_ENTITY_CTORS = {
    "cow": lambda world, pos, player: objects.Cow(world, pos),
    "zombie": lambda world, pos, player: objects.Zombie(world, pos, player),
    "skeleton": lambda world, pos, player: objects.Skeleton(world, pos, player),
}

# Feature name -> material laid down on the tile.
_FEATURE_MATERIALS = {
    "trees": "tree",
    "water": "water",
    "stone": "stone",
    "coal": "coal",
    "iron": "iron",
    "sand": "sand",
}

_WALKABLE = ("grass", "path", "sand")


# =============================================================================
#  World builder
# =============================================================================
class WorldBuilder:
    """Populates a fresh Crafter ``World`` from a :class:`WorldCfg`."""

    def __init__(self, world_cfg: WorldCfg):
        self._cfg = world_cfg

    def build(self, world, player) -> None:
        """Lay down terrain, features and entities. Player is already placed."""
        self._fill_base(world)
        self._place_features(world, player)
        self._place_entities(world, player)

    # -- terrain --------------------------------------------------------------
    def _fill_base(self, world) -> None:
        width, height = world.area
        base = self._cfg.base_terrain
        for x in range(width):
            for y in range(height):
                world[x, y] = base

    # -- features (materials) -------------------------------------------------
    def _place_features(self, world, player) -> None:
        start = tuple(player.pos)
        for name, spec in self._cfg.features.items():
            material = _FEATURE_MATERIALS.get(name)
            if material is None or not spec:
                continue
            tiles: list[tuple[int, int]] = []

            # Explicit positions.
            tiles.extend(tuple(p) for p in spec.get("positions", []) or [])

            # Rectangle (used for ponds).
            rect = spec.get("rect")
            if rect:
                rx, ry, rw, rh = rect
                for x in range(rx, rx + rw):
                    for y in range(ry, ry + rh):
                        tiles.append((x, y))

            # Random scatter.
            count = int(spec.get("count", 0) or 0)
            if count > 0:
                tiles.extend(self._random_free_tiles(world, start, count))

            for x, y in tiles:
                if not self._in_bounds(world, x, y):
                    LOG.warning("Feature '%s' tile (%s, %s) out of bounds.", name, x, y)
                    continue
                if (x, y) == start:
                    LOG.warning("Skipping '%s' on the player's start tile.", name)
                    continue
                world[x, y] = material

    # -- entities (objects) ---------------------------------------------------
    def _place_entities(self, world, player) -> None:
        start = tuple(player.pos)
        for name, spec in self._cfg.entities.items():
            ctor = _ENTITY_CTORS.get(name)
            if ctor is None or not spec:
                continue
            tiles: list[tuple[int, int]] = [tuple(p) for p in spec.get("positions", []) or []]

            count = int(spec.get("count", 0) or 0)
            if count > 0:
                tiles.extend(self._random_free_tiles(world, start, count))

            for x, y in tiles:
                if not self._can_hold_entity(world, x, y, start):
                    LOG.warning("Cannot place '%s' at (%s, %s); tile unavailable.", name, x, y)
                    continue
                world.add(ctor(world, (x, y), player))

    # -- helpers --------------------------------------------------------------
    def _random_free_tiles(self, world, start, count) -> list[tuple[int, int]]:
        width, height = world.area
        candidates = [
            (x, y)
            for x in range(width)
            for y in range(height)
            if (x, y) != start
            and world[x, y][1] is None
            and world[x, y][0] in _WALKABLE
        ]
        if not candidates:
            return []
        idx = world.random.choice(len(candidates), size=min(count, len(candidates)), replace=False)
        return [candidates[i] for i in np.atleast_1d(idx)]

    def _can_hold_entity(self, world, x, y, start) -> bool:
        if not self._in_bounds(world, x, y) or (x, y) == start:
            return False
        material, obj = world[x, y]
        return obj is None and material in _WALKABLE

    @staticmethod
    def _in_bounds(world, x, y) -> bool:
        width, height = world.area
        return 0 <= x < width and 0 <= y < height


# =============================================================================
#  Custom environment
# =============================================================================
class CustomCrafterEnv(crafter.Env):
    """
    Crafter environment with a deterministic, hand-built world.

    Keeps the real engine and achievement system; overrides only world
    generation, daylight and monster balancing.
    """

    def __init__(self, world_cfg: WorldCfg, seed: int = 0):
        width, height = world_cfg.size
        super().__init__(
            area=(width, height),
            view=(9, 9),
            size=(64, 64),
            reward=True,
            length=0,          # 0 = no internal episode length limit; we cap turns
            seed=seed,
        )
        self._world_cfg = world_cfg
        self._builder = WorldBuilder(world_cfg)
        self._world_seed = seed

    # -- deterministic control ------------------------------------------------
    def set_world_seed(self, seed: int) -> None:
        """Choose the seed used to build the next world (via ``reset``)."""
        self._world_seed = int(seed)

    def reset(self):
        """Build the custom world instead of Crafter's procedural one."""
        cfg = self._world_cfg
        self._episode += 1
        self._step = 0
        self._world.reset(seed=self._world_seed)
        self._update_time()

        start = tuple(cfg.player_start) if cfg.player_start else (
            self._world.area[0] // 2, self._world.area[1] // 2)
        self._player = objects.Player(self._world, start)
        self._apply_inventory_overrides()
        self._last_health = self._player.health
        self._world.add(self._player)

        self._unlocked = set()
        self._builder.build(self._world, self._player)
        return self._obs()

    def _apply_inventory_overrides(self) -> None:
        for item, amount in self._world_cfg.inventory.items():
            maximum = crafter.constants.items[item]["max"]
            self._player.inventory[item] = max(0, min(int(amount), maximum))

    # -- freeze the world -----------------------------------------------------
    def _update_time(self):
        if self._world_cfg.freeze_daylight:
            self._world.daylight = 1.0
        else:
            super()._update_time()

    def _balance_chunk(self, chunk, objs):
        if self._world_cfg.static:
            return  # no dynamic (de)spawning in a controlled world
        super()._balance_chunk(chunk, objs)

    # -- convenience accessors ------------------------------------------------
    @property
    def world(self):
        return self._world

    @property
    def player(self):
        return self._player

    @property
    def textures(self):
        return self._textures
