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

from perspective_test.archipelago import ArchipelagoMap


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

"""
Test suite for ship.py.

Run with:
    pytest test_ship.py -v
"""

import numpy as np
import pytest

from perspective_test.archipelago import ArchipelagoMap
from perspective_test.ship import Ship


def make_open_water_map(size=10):
    """Builds a map that is entirely open water, with a fixed, known start
    and goal. Using an all-water map (rather than a randomly generated one)
    means these tests are checking Ship's movement/rotation logic in
    isolation, without depending on where any particular seed happens to
    put land."""
    archipelago_map = ArchipelagoMap(seed=1)
    archipelago_map.size = size
    archipelago_map.terrain_grid = np.full((size, size), "water", dtype=object)
    archipelago_map.ship_start_position = (5, 5)
    archipelago_map.ship_goal_position = (0, 0)
    return archipelago_map


# ----------------------------------------------------------------------
# Turning
# ----------------------------------------------------------------------


def test_four_right_turns_return_to_original_heading():
    ship = Ship(make_open_water_map(), starting_heading="north")
    for _ in range(4):
        ship.turn_right()
    assert ship.heading == "north"


def test_four_left_turns_return_to_original_heading():
    ship = Ship(make_open_water_map(), starting_heading="north")
    for _ in range(4):
        ship.turn_left()
    assert ship.heading == "north"


def test_turning_right_cycles_clockwise():
    ship = Ship(make_open_water_map(), starting_heading="north")
    ship.turn_right()
    assert ship.heading == "east"
    ship.turn_right()
    assert ship.heading == "south"
    ship.turn_right()
    assert ship.heading == "west"


def test_turning_does_not_change_position():
    ship = Ship(make_open_water_map(), starting_heading="north")
    position_before = ship.position
    ship.turn_left()
    ship.turn_right()
    ship.turn_right()
    assert ship.position == position_before


# ----------------------------------------------------------------------
# Moving
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "heading, expected_position_after_forward",
    [
        ("north", (4, 5)),
        ("south", (6, 5)),
        ("east", (5, 6)),
        ("west", (5, 4)),
    ],
)
def test_move_forward_steps_one_tile_in_facing_direction(heading, expected_position_after_forward):
    ship = Ship(make_open_water_map(), starting_heading=heading)
    move_succeeded = ship.move_forward()

    assert move_succeeded
    assert ship.position == expected_position_after_forward


def test_move_backward_steps_opposite_the_facing_direction_without_turning():
    ship = Ship(make_open_water_map(), starting_heading="north")
    ship.move_backward()

    assert ship.position == (6, 5)  # south of start, since facing north
    assert ship.heading == "north"  # backing up does not turn the ship around


def test_move_forward_then_backward_returns_to_start():
    ship = Ship(make_open_water_map(), starting_heading="east")
    starting_position = ship.position
    ship.move_forward()
    ship.move_backward()
    assert ship.position == starting_position


def test_move_is_blocked_by_land():
    archipelago_map = make_open_water_map()
    archipelago_map.terrain_grid[4, 5] = "land"  # directly north of the ship
    ship = Ship(archipelago_map, starting_heading="north")

    move_succeeded = ship.move_forward()

    assert not move_succeeded
    assert ship.position == (5, 5)  # unchanged


def test_move_is_blocked_at_the_edge_of_the_map():
    archipelago_map = make_open_water_map(size=10)
    archipelago_map.ship_start_position = (0, 0)
    ship = Ship(archipelago_map, starting_heading="north")

    move_succeeded = ship.move_forward()

    assert not move_succeeded
    assert ship.position == (0, 0)


def test_move_history_is_recorded():
    ship = Ship(make_open_water_map(), starting_heading="north")
    ship.move_forward()
    ship.turn_right()
    ship.move_forward()
    assert ship.move_history == ["forward", "turn_right", "forward"]


# ----------------------------------------------------------------------
# Egocentric local view
# ----------------------------------------------------------------------


def test_local_view_is_five_rows_by_three_columns():
    ship = Ship(make_open_water_map())
    view = ship.get_local_view()
    assert len(view) == 5
    assert all(len(row) == 3 for row in view)


def test_ship_is_always_at_the_center_of_its_own_view():
    """Regardless of heading, the ship should see itself at local position
    (row 3, col 1) - three rows down (since forward extends 3 rows above
    it) and in the middle column."""
    for heading in Ship.HEADINGS_CLOCKWISE:
        ship = Ship(make_open_water_map(), starting_heading=heading)
        view = ship.get_local_view()
        assert view[3][1] == "ship"


def test_local_view_matches_world_map_when_facing_north():
    archipelago_map = make_open_water_map()
    # Place a distinctive tile two rows ahead of the ship (north = up) so we
    # can confirm it shows up in the correct spot in the egocentric view.
    archipelago_map.terrain_grid[3, 5] = "land"
    ship = Ship(archipelago_map, starting_heading="north")

    view = ship.get_local_view()

    # Two tiles ahead should be row index 1 (since row 0 is 3 tiles ahead,
    # row 1 is 2 tiles ahead), center column.
    assert view[1][1] == "land"


def test_local_view_rotates_so_forward_is_always_up():
    """Put a distinctive tile at a fixed world position, then check that
    whichever direction the ship faces toward it, that tile appears
    'two rows ahead' (index 1) in the egocentric view - proving the view
    actually rotates with the ship instead of staying north-up."""
    archipelago_map = make_open_water_map()
    ship_row, ship_col = 5, 5
    archipelago_map.ship_start_position = (ship_row, ship_col)

    headings_and_the_world_tile_two_ahead = {
        "north": (ship_row - 2, ship_col),
        "south": (ship_row + 2, ship_col),
        "east": (ship_row, ship_col + 2),
        "west": (ship_row, ship_col - 2),
    }

    for heading, tile_two_ahead in headings_and_the_world_tile_two_ahead.items():
        archipelago_map.terrain_grid = np.full((10, 10), "water", dtype=object)
        archipelago_map.terrain_grid[tile_two_ahead] = "land"

        ship = Ship(archipelago_map, starting_heading=heading)
        view = ship.get_local_view()

        assert view[1][1] == "land", f"failed for heading={heading}"


def test_unknown_tiles_shown_for_out_of_bounds_positions():
    archipelago_map = make_open_water_map()
    archipelago_map.ship_start_position = (0, 0)
    ship = Ship(archipelago_map, starting_heading="north")

    view = ship.get_local_view()

    # Every tile "ahead" of the ship here is off the top edge of the map.
    assert view[0] == ["unknown", "unknown", "unknown"]
    assert view[1] == ["unknown", "unknown", "unknown"]
    assert view[2] == ["unknown", "unknown", "unknown"]


def test_goal_tile_is_labeled_goal_not_its_terrain_type():
    archipelago_map = make_open_water_map()
    archipelago_map.ship_start_position = (5, 5)
    archipelago_map.ship_goal_position = (4, 5)  # one tile north of the ship
    ship = Ship(archipelago_map, starting_heading="north")

    view = ship.get_local_view()

    assert view[2][1] == "goal"


# ----------------------------------------------------------------------
# The full map view must never reveal the ship's position
# ----------------------------------------------------------------------


def test_full_map_render_never_shows_ship_position(capsys):
    """Renders the full map, then moves the ship onto a tile that is
    visually distinctive (all water elsewhere) and confirms the printed
    output doesn't change - i.e. the map view is blind to where the ship
    actually is."""
    archipelago_map = make_open_water_map()
    archipelago_map.render(use_emoji=False)
    map_output_before = capsys.readouterr().out

    ship = Ship(archipelago_map, starting_heading="north")
    ship.move_forward()
    ship.move_forward()

    archipelago_map.render(use_emoji=False)
    map_output_after = capsys.readouterr().out

    assert map_output_before == map_output_after
    assert "S" not in map_output_after  # no ship marker character anywhere