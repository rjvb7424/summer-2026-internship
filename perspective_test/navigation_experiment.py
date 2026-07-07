"""
Core logic for one trial of the archipelago navigation experiment: generate
a map, give the solver the full map (but not its own position), then
repeatedly show it its local egocentric view and ask for the next move,
until it reaches the goal or runs out of turns.
"""
import re

from archipelago import ArchipelagoMap
from ship import Ship

# --------------------------------------------------------------------------
# Constants - change these to adjust how a trial is scored.
# --------------------------------------------------------------------------

# The moves a solver is allowed to issue. These map directly onto Ship's
# egocentric movement methods.
VALID_MOVES = ("FORWARD", "BACKWARD", "LEFT", "RIGHT")

# When no explicit max_turns is passed to NavigationTrial, the turn budget
# is set to (Manhattan distance from start to goal) * DEFAULT_TURNS_PER_TILE,
# with a floor of MINIMUM_MAX_TURNS. This scales the grace period with how
# far apart start and goal actually are, while still giving short trips a
# reasonable number of turns to recover from early mistakes.
DEFAULT_TURNS_PER_TILE = 4
MINIMUM_MAX_TURNS = 20

# Single-letter symbols used when rendering terrain into the text prompt
# sent to the solver. Kept separate from Ship/ArchipelagoMap's own emoji
# rendering, since plain ASCII is cheaper and less ambiguous for an LLM
# prompt than emoji.
TERRAIN_LETTERS = {
    "water": "W",
    "beach": "B",
    "land": "L",
    "mountain": "M",
}
LOCAL_VIEW_LETTERS = {
    **TERRAIN_LETTERS,
    "unknown": "?",
    "ship": "S",
    "goal": "G",
}


def extract_move(response_text):
    """Extracts the solver's intended move (FORWARD/BACKWARD/LEFT/RIGHT)
    from its raw response text. Returns None if no valid move is found.

    Checks explicit "final answer" style patterns first (useful for
    reasoning models that think out loud before concluding), then falls
    back to the last standalone occurrence of a valid move word anywhere
    in the response.
    """
    text = response_text.strip()
    move_pattern = "|".join(VALID_MOVES)

    final_answer_patterns = [
        rf"\\boxed\{{({move_pattern})\}}",
        rf"final answer[^A-Za-z]{{0,20}}({move_pattern})\b",
        rf"answer is[:\s]*({move_pattern})\b",
    ]
    for pattern in final_answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).upper()

    matches = re.findall(rf"\b({move_pattern})\b", text.upper())
    return matches[-1] if matches else None


class NavigationTrial:
    """One full run of the navigation experiment: one generated map, one
    ship, one sequence of solver calls until the ship reaches the goal or
    the turn budget runs out."""

    def __init__(self, map_size=10, max_turns=None, seed=None):
        self.archipelago_map = ArchipelagoMap(size=map_size, seed=seed)
        self.ship = Ship(self.archipelago_map)
        self.max_turns = max_turns if max_turns is not None else self._default_max_turns()

        # Every turn's prompt, response, and outcome, in order - this is
        # both how history is fed back into later prompts (since each
        # solver call is a fresh one-shot call with no memory of its own)
        # and the detailed transcript saved to the results file.
        self.turn_log = []

        # Built once, since the map itself never changes over the course
        # of a trial - only the ship's position and heading do.
        self._map_and_instructions_prompt = self._build_initial_instructions()

    def _default_max_turns(self):
        start_row, start_col = self.ship.position
        goal_row, goal_col = self.archipelago_map.ship_goal_position
        manhattan_distance = abs(start_row - goal_row) + abs(start_col - goal_col)
        return max(MINIMUM_MAX_TURNS, manhattan_distance * DEFAULT_TURNS_PER_TILE)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _render_full_map_as_text(self):
        """Renders the full map as plain-letter rows, with the goal marked
        but the ship's position deliberately omitted."""
        map_rows = []
        for row in range(self.archipelago_map.size):
            row_letters = []
            for col in range(self.archipelago_map.size):
                if (row, col) == self.archipelago_map.ship_goal_position:
                    row_letters.append("G")
                else:
                    terrain_type = self.archipelago_map.terrain_grid[row, col]
                    row_letters.append(TERRAIN_LETTERS[terrain_type])
            map_rows.append("".join(row_letters))
        return "\n".join(map_rows)

    def _render_local_view_as_text(self):
        """Renders the ship's current egocentric view as plain-letter rows."""
        view_grid = self.ship.get_local_view()
        return "\n".join(
            "".join(LOCAL_VIEW_LETTERS[tile] for tile in view_row) for view_row in view_grid
        )

    def _build_initial_instructions(self):
        return (
            "You are navigating a ship across an archipelago to reach a goal.\n"
            f"The full map is {self.archipelago_map.size}x{self.archipelago_map.size} tiles, "
            "shown below.\n"
            "W = open water (sailable). B = beach/shallow water (NOT sailable). "
            "L = land (NOT sailable). M = mountain (NOT sailable). "
            "G = the goal you must reach.\n\n"
            f"{self._render_full_map_as_text()}\n\n"
            "You do NOT know your own position on this map. Each turn you will "
            "be shown only a small local view of the tiles immediately around "
            "your ship, from your own point of view: forward is always the "
            "TOP row of that view, and left/right are always the left/right "
            "columns, regardless of which way you actually happen to be "
            "facing on the full map above. Use the local view together with "
            "the full map to work out where you are and steer toward the "
            "goal.\n\n"
            "On every turn, respond with exactly one move as the very last "
            "word of your response: FORWARD, BACKWARD, LEFT, or RIGHT. "
            "FORWARD/BACKWARD move you one tile in the direction you're "
            "currently facing (a move into land, beach, mountain, or off the "
            "edge of the map fails and you stay in place). LEFT/RIGHT rotate "
            "you 90 degrees without changing your position."
        )

    def _build_prompt_for_turn(self, turn_number, current_view_text):
        sections = [self._map_and_instructions_prompt]

        if self.turn_log:
            history_lines = ["History so far:"]
            for past_turn in self.turn_log:
                chosen_move = past_turn["parsed_move"] or "an unrecognized move"
                outcome = "succeeded" if past_turn["move_valid"] else "was blocked or invalid"
                history_lines.append(
                    f"Turn {past_turn['turn']} - local view:\n{past_turn['local_view']}\n"
                    f"You chose {chosen_move}, which {outcome}."
                )
            sections.append("\n\n".join(history_lines))

        sections.append(
            f"Turn {turn_number}. Your current local view (forward is the top row):\n"
            f"{current_view_text}\n\n"
            "What is your next move?"
        )
        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Running the trial
    # ------------------------------------------------------------------

    def _apply_move(self, parsed_move):
        """Applies a parsed move to the ship. Returns True if it was a
        recognized move that actually succeeded, False otherwise (an
        unrecognized move, or a recognized move that was blocked)."""
        if parsed_move == "FORWARD":
            return self.ship.move_forward()
        if parsed_move == "BACKWARD":
            return self.ship.move_backward()
        if parsed_move == "LEFT":
            self.ship.turn_left()
            return True
        if parsed_move == "RIGHT":
            self.ship.turn_right()
            return True
        return False

    def run(self, solver):
        """Runs the trial to completion by repeatedly calling solver(prompt)
        and applying the move it returns, until the ship reaches the goal
        or the turn budget is used up. Returns a JSON-serializable result
        dict summarizing the whole trial."""
        turn_number = 0

        while turn_number < self.max_turns:
            if self.ship.position == self.archipelago_map.ship_goal_position:
                break

            turn_number += 1
            current_view_text = self._render_local_view_as_text()
            prompt = self._build_prompt_for_turn(turn_number, current_view_text)
            solver_result = solver(prompt)

            turn_record = {
                "turn": turn_number,
                "position_before": list(self.ship.position),
                "heading_before": self.ship.heading,
                "local_view": current_view_text,
                "prompt": prompt,
            }

            if solver_result is None:
                # The solver already exhausted its own internal retries -
                # this is an unrecoverable error for this turn, so stop the
                # trial here rather than pretending nothing happened.
                turn_record.update({
                    "raw_response": None,
                    "parsed_move": None,
                    "move_valid": False,
                    "error": "solver_call_failed",
                })
                self.turn_log.append(turn_record)
                break

            parsed_move = extract_move(solver_result["text"])
            move_succeeded = self._apply_move(parsed_move)

            turn_record.update({
                "raw_response": solver_result["text"],
                "parsed_move": parsed_move,
                "move_valid": move_succeeded,
                "position_after": list(self.ship.position),
                "heading_after": self.ship.heading,
                "elapsed_seconds": solver_result.get("elapsed_seconds"),
                "prompt_tokens": solver_result.get("prompt_tokens"),
                "output_tokens": solver_result.get("output_tokens"),
                "thinking_tokens": solver_result.get("thinking_tokens"),
                "total_tokens": solver_result.get("total_tokens"),
            })
            self.turn_log.append(turn_record)

        reached_goal = self.ship.position == self.archipelago_map.ship_goal_position
        return self._build_result(reached_goal, turn_number)

    def _build_result(self, reached_goal, turns_taken):
        def total(field_name):
            return sum(turn.get(field_name) or 0 for turn in self.turn_log)

        final_row, final_col = self.ship.position
        goal_row, goal_col = self.archipelago_map.ship_goal_position

        return {
            "map_size": self.archipelago_map.size,
            "map_seed": self.archipelago_map.seed,
            "start_position": list(self.archipelago_map.ship_start_position),
            "goal_position": list(self.archipelago_map.ship_goal_position),
            "max_turns": self.max_turns,
            "turns_taken": turns_taken,
            "reached_goal": reached_goal,
            "timed_out": (not reached_goal) and turns_taken >= self.max_turns,
            "final_position": list(self.ship.position),
            "final_manhattan_distance_to_goal": abs(final_row - goal_row) + abs(final_col - goal_col),
            "move_history": list(self.ship.move_history),
            "turn_log": self.turn_log,
            "total_elapsed_seconds": total("elapsed_seconds"),
            "total_prompt_tokens": total("prompt_tokens"),
            "total_output_tokens": total("output_tokens"),
            "total_thinking_tokens": total("thinking_tokens"),
            "total_tokens": total("total_tokens"),
        }