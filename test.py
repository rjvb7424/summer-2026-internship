"""
Test suite for archipelago.py.

Run with:
    pytest test_archipelago.py -v

Or just:
    pytest
(pytest auto-discovers any file named test_*.py or *_test.py)
"""

import numpy as np
import pytest

from archipelago import ArchipelagoMap


# A test function is just a regular function whose name starts with `test_`.
# pytest finds every one of these automatically - there's no registration
# step and no test class required (though you can group with classes if you
# want, similar to a JUnit @Test-annotated class).


def test_start_and_goal_are_on_navigable_terrain():
    """The ship's start and goal tiles must themselves be sailable water,
    not land - otherwise the ship couldn't even begin or end its voyage."""
    archipelago_map = ArchipelagoMap(seed=1)

    start_terrain = archipelago_map.terrain_grid[archipelago_map.ship_start_position]
    goal_terrain = archipelago_map.terrain_grid[archipelago_map.ship_goal_position]

    assert start_terrain in ArchipelagoMap.NAVIGABLE_TERRAIN_TYPES
    assert goal_terrain in ArchipelagoMap.NAVIGABLE_TERRAIN_TYPES


def test_start_and_goal_are_connected_by_water():
    """The core guarantee: a ship must always be able to sail from start to
    goal using only navigable tiles."""
    archipelago_map = ArchipelagoMap(seed=1)

    assert archipelago_map.path_exists(
        archipelago_map.ship_start_position, archipelago_map.ship_goal_position
    )


@pytest.mark.parametrize("seed", range(300))
def test_reachability_holds_across_many_random_maps(seed):
    """Runs the same reachability check across 300 different seeds. This is
    the important one: a single passing seed doesn't prove much, but 300
    passing seeds gives real confidence the guarantee holds in general.

    pytest.mark.parametrize re-runs this test once per seed and reports each
    one as a separate pass/fail, so if seed 217 ever fails you'll see exactly
    that in the output instead of one big generic failure.
    """
    archipelago_map = ArchipelagoMap(seed=seed)

    assert archipelago_map.path_exists(
        archipelago_map.ship_start_position, archipelago_map.ship_goal_position
    )


def test_start_and_goal_are_different_tiles():
    """A start position equal to the goal position would make for a
    trivial, zero-length voyage - that would be a bug, not a valid map."""
    archipelago_map = ArchipelagoMap(seed=1)

    assert archipelago_map.ship_start_position != archipelago_map.ship_goal_position


def test_same_seed_produces_identical_map():
    """Passing an explicit seed should be fully reproducible - useful for
    debugging a specific map or replaying a run from your benchmark logs."""
    first_map = ArchipelagoMap(seed=42)
    second_map = ArchipelagoMap(seed=42)

    assert np.array_equal(first_map.terrain_grid, second_map.terrain_grid)
    assert first_map.ship_start_position == second_map.ship_start_position
    assert first_map.ship_goal_position == second_map.ship_goal_position


def test_no_seed_produces_a_different_map_each_time():
    """Leaving seed=None (the default) should give you a fresh map every
    call, which is what makes each benchmark run a new test case."""
    first_map = ArchipelagoMap()
    second_map = ArchipelagoMap()

    assert first_map.seed != second_map.seed


def test_map_is_square_and_matches_requested_size():
    requested_size = 25
    archipelago_map = ArchipelagoMap(size=requested_size, seed=1)

    assert archipelago_map.terrain_grid.shape == (requested_size, requested_size)


def test_open_water_meets_the_minimum_fraction_requirement():
    """The generator is supposed to reject (and silently regenerate) any map
    whose largest connected body of water is too small. This test checks
    that guarantee held for the map we actually got back."""
    minimum_fraction = 0.35
    archipelago_map = ArchipelagoMap(seed=1, minimum_open_water_fraction=minimum_fraction)

    largest_region = archipelago_map._find_largest_connected_water_region()
    map_area = archipelago_map.size * archipelago_map.size

    assert len(largest_region) >= minimum_fraction * map_area


def test_path_exists_returns_false_for_a_genuinely_unreachable_pair():
    """path_exists() should correctly say NO when there truly isn't a route,
    not just always return True. We fake this by asking about two tiles we
    know are on opposite sides of solid land with no water gap: a land tile
    can't be part of any water path, so asking to reach it should fail."""
    archipelago_map = ArchipelagoMap(seed=1)

    land_positions = np.argwhere(archipelago_map.terrain_grid == "land")
    if len(land_positions) == 0:
        pytest.skip("this particular seed produced no land tiles to test against")

    unreachable_target = tuple(land_positions[0])
    assert not archipelago_map.path_exists(
        archipelago_map.ship_start_position, unreachable_target
    )
