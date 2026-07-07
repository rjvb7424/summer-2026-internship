"""
Test suite for navigation_experiment.py.

Run with:
    pytest test_navigation_experiment.py -v

These tests never call a real AI model - they use small stub "solver"
functions (plain Python functions matching the solver(prompt) -> dict|None
interface) to drive NavigationTrial deterministically, so the experiment
harness itself (prompt building, move parsing, turn budget, result shape)
can be verified without needing network access or an actual model.
"""
import json

import numpy as np
import pytest

from archipelago import ArchipelagoMap
from navigation_experiment import NavigationTrial, extract_move


def make_solver_response(text, **overrides):
    """Builds a fake solver return value with the same shape call_gpt /
    call_deepseek would return."""
    response = {
        "text": text,
        "elapsed_seconds": 1.0,
        "prompt_tokens": 10,
        "output_tokens": 5,
        "thinking_tokens": 0,
        "total_tokens": 15,
    }
    response.update(overrides)
    return response


def make_open_water_trial(size=10, start=(5, 5), goal=(5, 8), max_turns=20):
    """An all-water map with a known start/goal, so tests can predict
    exactly how many moves are needed instead of depending on a randomly
    generated map's layout."""
    trial = NavigationTrial(map_size=size, seed=1, max_turns=max_turns)
    trial.archipelago_map.terrain_grid = np.full((size, size), "water", dtype=object)
    trial.archipelago_map.ship_start_position = start
    trial.archipelago_map.ship_goal_position = goal
    trial.ship.position = start
    trial.ship.heading = "north"
    # The initial instructions prompt embeds the map as it existed at
    # __init__ time, before this override - rebuild it against the new,
    # all-water map so prompts reflect what the test actually set up.
    trial._map_and_instructions_prompt = trial._build_initial_instructions()
    return trial


# ----------------------------------------------------------------------
# extract_move
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "response_text, expected_move",
    [
        ("FORWARD", "FORWARD"),
        ("I think I should go left. LEFT", "LEFT"),
        ("final answer: RIGHT", "RIGHT"),
        (r"\boxed{BACKWARD}", "BACKWARD"),
        ("Let me reason step by step... forward seems right so FORWARD", "FORWARD"),
    ],
)
def test_extract_move_finds_the_intended_move(response_text, expected_move):
    assert extract_move(response_text) == expected_move


def test_extract_move_returns_none_when_no_valid_move_present():
    assert extract_move("I'm not sure what to do here.") is None


def test_extract_move_is_case_insensitive():
    assert extract_move("i'll go forward") == "FORWARD"


# ----------------------------------------------------------------------
# A trial that succeeds
# ----------------------------------------------------------------------


def test_trial_reaches_goal_when_solver_moves_correctly():
    """Goal is 3 tiles east of a north-facing ship: turn right, then move
    forward 3 times."""
    trial = make_open_water_trial(start=(5, 5), goal=(5, 8))
    scripted_moves = iter(["RIGHT", "FORWARD", "FORWARD", "FORWARD"])

    def scripted_solver(prompt):
        return make_solver_response(next(scripted_moves))

    result = trial.run(scripted_solver)

    assert result["reached_goal"] is True
    assert result["timed_out"] is False
    assert result["turns_taken"] == 4
    assert result["final_position"] == [5, 8]
    assert result["final_manhattan_distance_to_goal"] == 0


def test_trial_stops_calling_the_solver_once_the_goal_is_reached():
    """The turn where the ship arrives should be the last call - the
    solver should not be asked for a move after already being at the goal."""
    trial = make_open_water_trial(start=(5, 5), goal=(5, 6))
    call_count = 0

    def counting_solver(prompt):
        nonlocal call_count
        call_count += 1
        return make_solver_response("RIGHT" if call_count == 1 else "FORWARD")

    result = trial.run(counting_solver)

    assert result["reached_goal"] is True
    assert call_count == 2
    assert result["turns_taken"] == 2


# ----------------------------------------------------------------------
# A trial that times out
# ----------------------------------------------------------------------


def test_trial_times_out_if_solver_never_reaches_the_goal():
    trial = make_open_water_trial(start=(5, 5), goal=(5, 8), max_turns=5)

    def stuck_solver(prompt):
        return make_solver_response("LEFT")  # spins in place forever

    result = trial.run(stuck_solver)

    assert result["reached_goal"] is False
    assert result["timed_out"] is True
    assert result["turns_taken"] == 5


# ----------------------------------------------------------------------
# Blocked and unparseable moves still consume a turn
# ----------------------------------------------------------------------


def test_move_into_land_is_recorded_as_blocked_but_still_counts_as_a_turn():
    trial = make_open_water_trial(start=(5, 5), goal=(5, 8), max_turns=5)
    trial.archipelago_map.terrain_grid[4, 5] = "land"  # directly north

    def forward_into_land_solver(prompt):
        return make_solver_response("FORWARD")

    result = trial.run(forward_into_land_solver)

    assert result["timed_out"] is True
    assert result["turns_taken"] == 5
    assert result["turn_log"][0]["move_valid"] is False
    assert result["turn_log"][0]["position_after"] == [5, 5]  # never moved


def test_unparseable_response_is_recorded_and_still_consumes_a_turn():
    trial = make_open_water_trial(start=(5, 5), goal=(5, 8), max_turns=3)

    def confused_solver(prompt):
        return make_solver_response("I have no idea what to do.")

    result = trial.run(confused_solver)

    assert result["turns_taken"] == 3
    assert all(turn["parsed_move"] is None for turn in result["turn_log"])
    assert all(turn["move_valid"] is False for turn in result["turn_log"])


# ----------------------------------------------------------------------
# Solver failure aborts the trial early
# ----------------------------------------------------------------------


def test_solver_returning_none_aborts_the_trial():
    trial = make_open_water_trial(start=(5, 5), goal=(5, 8), max_turns=20)

    def failing_solver(prompt):
        return None

    result = trial.run(failing_solver)

    assert result["reached_goal"] is False
    assert result["turns_taken"] == 1
    assert result["turn_log"][0]["error"] == "solver_call_failed"


# ----------------------------------------------------------------------
# History is fed back into later prompts
# ----------------------------------------------------------------------


def test_later_prompts_include_earlier_turns_history():
    trial = make_open_water_trial(start=(5, 5), goal=(5, 8), max_turns=3)
    scripted_moves = iter(["RIGHT", "FORWARD", "FORWARD"])
    captured_prompts = []

    def recording_solver(prompt):
        captured_prompts.append(prompt)
        return make_solver_response(next(scripted_moves))

    trial.run(recording_solver)

    assert "History so far" not in captured_prompts[0]
    assert "History so far" in captured_prompts[1]
    assert "You chose RIGHT" in captured_prompts[1]
    assert "You chose RIGHT" in captured_prompts[2]
    assert "You chose FORWARD" in captured_prompts[2]


# ----------------------------------------------------------------------
# Token/time aggregation
# ----------------------------------------------------------------------


def test_token_and_time_totals_are_summed_across_turns():
    trial = make_open_water_trial(start=(5, 5), goal=(5, 7), max_turns=5)
    scripted_moves = iter(["RIGHT", "FORWARD", "FORWARD"])

    def scripted_solver(prompt):
        return make_solver_response(
            next(scripted_moves), prompt_tokens=100, output_tokens=20, thinking_tokens=50, total_tokens=170
        )

    result = trial.run(scripted_solver)

    assert result["total_prompt_tokens"] == 300
    assert result["total_output_tokens"] == 60
    assert result["total_thinking_tokens"] == 150
    assert result["total_tokens"] == 510
    assert result["total_elapsed_seconds"] == pytest.approx(3.0)


# ----------------------------------------------------------------------
# The default turn budget scales with distance
# ----------------------------------------------------------------------


def test_default_max_turns_scales_with_distance_to_goal():
    close_trial = make_open_water_trial(start=(5, 5), goal=(5, 6), max_turns=None)
    close_trial.max_turns = close_trial._default_max_turns()

    far_trial = make_open_water_trial(start=(0, 0), goal=(9, 9), max_turns=None)
    far_trial.max_turns = far_trial._default_max_turns()

    assert far_trial.max_turns > close_trial.max_turns


def test_default_max_turns_has_a_floor_for_very_short_trips():
    from navigation_experiment import MINIMUM_MAX_TURNS

    trial = make_open_water_trial(start=(5, 5), goal=(5, 6))
    assert trial._default_max_turns() >= MINIMUM_MAX_TURNS


# ----------------------------------------------------------------------
# The result must be JSON-serializable, since that's the whole point
# ----------------------------------------------------------------------


def test_result_is_json_serializable():
    trial = make_open_water_trial(start=(5, 5), goal=(5, 6), max_turns=5)

    def scripted_solver(prompt):
        return make_solver_response("RIGHT")

    result = trial.run(scripted_solver)

    # Should not raise.
    serialized = json.dumps(result)
    assert isinstance(serialized, str)


def test_real_generated_map_can_also_complete_a_trial_via_bfs_shortest_path():
    """Sanity check against an actual (non-rigged) generated map: drive the
    ship using a real BFS shortest path translated into egocentric moves,
    and confirm the trial framework reports a success. This guards against
    the rigged all-water tests accidentally hiding a bug that only shows up
    with real land/beach obstacles."""
    from collections import deque

    trial = NavigationTrial(map_size=10, seed=3)
    archipelago_map = trial.archipelago_map

    # BFS shortest path over navigable tiles, start to goal.
    start = archipelago_map.ship_start_position
    goal = archipelago_map.ship_goal_position
    previous = {start: None}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for neighbor in archipelago_map._get_neighbors(*current):
            if neighbor not in previous and archipelago_map.terrain_grid[neighbor] in (
                ArchipelagoMap.NAVIGABLE_TERRAIN_TYPES
            ):
                previous[neighbor] = current
                queue.append(neighbor)

    path = [goal]
    while previous[path[-1]] is not None:
        path.append(previous[path[-1]])
    path.reverse()

    # Precompute the full list of egocentric commands needed to walk this
    # path, by simulating a virtual heading alongside the real ship's
    # starting heading (rather than deciding turns reactively inside the
    # solver callback, which would need to peek/re-queue and is easy to
    # get subtly wrong).
    heading_for_step = {(-1, 0): "north", (1, 0): "south", (0, 1): "east", (0, -1): "west"}
    headings_clockwise = trial.ship.HEADINGS_CLOCKWISE
    virtual_heading = trial.ship.heading
    commands = []

    for step_index in range(1, len(path)):
        row_delta = path[step_index][0] - path[step_index - 1][0]
        col_delta = path[step_index][1] - path[step_index - 1][1]
        needed_heading = heading_for_step[(row_delta, col_delta)]

        current_index = headings_clockwise.index(virtual_heading)
        target_index = headings_clockwise.index(needed_heading)
        clockwise_steps = (target_index - current_index) % 4

        if clockwise_steps == 1:
            commands.append("RIGHT")
        elif clockwise_steps == 2:
            commands.extend(["RIGHT", "RIGHT"])
        elif clockwise_steps == 3:
            commands.append("LEFT")
        # clockwise_steps == 0 means already facing the right way: no turn needed.

        virtual_heading = needed_heading
        commands.append("FORWARD")

    scripted_moves = iter(commands)

    def path_following_solver(prompt):
        return make_solver_response(next(scripted_moves))

    result = trial.run(path_following_solver)
    assert result["reached_goal"] is True