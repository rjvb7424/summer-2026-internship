from legacy.pt_test.archipelago import ArchipelagoMap


class Ship:
    """A ship that moves egocentrically (forward/backward/turn left/turn
    right, relative to its own facing) around an ArchipelagoMap.

    The ship only ever sees a small window of tiles around itself (see
    render_local_view), never the whole map - it's up to whoever is
    steering the ship to figure out where they are by comparing that local
    window against the full map layout.
    """

    # The four headings the ship can face, in clockwise order. Turning right
    # moves one step forward through this list; turning left moves one step
    # backward through it.
    HEADINGS_CLOCKWISE = ["north", "east", "south", "west"]

    # Which way is "forward" on the grid for each heading. Grid rows increase
    # downward and columns increase rightward, so "north" is a decreasing row.
    FORWARD_VECTOR_BY_HEADING = {
        "north": (-1, 0),
        "east": (0, 1),
        "south": (1, 0),
        "west": (0, -1),
    }

    # How far the ship can see: one tile in every direction, plus this many
    # extra tiles straight ahead in the direction it's facing.
    SIDEWAYS_VISIBILITY_RANGE = 1
    EXTRA_FORWARD_VISIBILITY_RANGE = 2

    def __init__(self, archipelago_map, starting_heading="north"):
        self.archipelago_map = archipelago_map
        self.position = archipelago_map.ship_start_position
        self.heading = starting_heading
        # A log of every move attempt, useful for debugging a solver's
        # behavior after the fact.
        self.move_history = []

    # ------------------------------------------------------------------
    # Turning
    # ------------------------------------------------------------------

    def face(self, heading):
        """Turns the ship to face a specific compass direction directly
        (however many 90-degree turns that takes), without moving. Useful
        as a shortcut instead of chaining several turn_left/turn_right
        calls to reach a specific heading."""
        if heading not in self.HEADINGS_CLOCKWISE:
            raise ValueError(f"Unknown heading: {heading!r}")
        self.heading = heading
        self.move_history.append(f"face_{heading}")

    def turn_left(self):
        """Rotates the ship 90 degrees counter-clockwise. Turning doesn't
        change position, so it always succeeds."""
        current_index = self.HEADINGS_CLOCKWISE.index(self.heading)
        new_index = (current_index - 1) % len(self.HEADINGS_CLOCKWISE)
        self.heading = self.HEADINGS_CLOCKWISE[new_index]
        self.move_history.append("turn_left")

    def turn_right(self):
        """Rotates the ship 90 degrees clockwise. Turning doesn't change
        position, so it always succeeds."""
        current_index = self.HEADINGS_CLOCKWISE.index(self.heading)
        new_index = (current_index + 1) % len(self.HEADINGS_CLOCKWISE)
        self.heading = self.HEADINGS_CLOCKWISE[new_index]
        self.move_history.append("turn_right")

    def _get_right_vector_for_heading(self, heading):
        """The direction that is 90 degrees clockwise from the given
        heading - i.e. what "to the right" means when facing that way."""
        current_index = self.HEADINGS_CLOCKWISE.index(heading)
        right_heading = self.HEADINGS_CLOCKWISE[(current_index + 1) % 4]
        return self.FORWARD_VECTOR_BY_HEADING[right_heading]

    # ------------------------------------------------------------------
    # Moving
    # ------------------------------------------------------------------

    def move_forward(self):
        """Attempts to move the ship one tile in the direction it's facing.
        Returns True if the move succeeded, False if it was blocked (either
        by the edge of the map or by non-navigable terrain like land)."""
        return self._attempt_move(self.FORWARD_VECTOR_BY_HEADING[self.heading], "forward")

    def move_backward(self):
        """Attempts to move the ship one tile opposite the direction it's
        facing (without turning around first). Returns True if the move
        succeeded, False if it was blocked."""
        forward_row, forward_col = self.FORWARD_VECTOR_BY_HEADING[self.heading]
        return self._attempt_move((-forward_row, -forward_col), "backward")

    def _attempt_move(self, direction_vector, move_label):
        row_delta, col_delta = direction_vector
        current_row, current_col = self.position
        target_position = (current_row + row_delta, current_col + col_delta)

        if self._is_navigable(target_position):
            self.position = target_position
            self.move_history.append(move_label)
            return True

        self.move_history.append(f"{move_label} (blocked)")
        return False

    def _is_navigable(self, position):
        """True if position is on the map and is a terrain type the ship
        can sail through."""
        row, col = position
        map_size = self.archipelago_map.size
        if not (0 <= row < map_size and 0 <= col < map_size):
            return False
        return self.archipelago_map.terrain_grid[row, col] in (
            ArchipelagoMap.NAVIGABLE_TERRAIN_TYPES
        )

    # ------------------------------------------------------------------
    # Egocentric local view
    # ------------------------------------------------------------------

    def get_local_view(self):
        """Returns the ship's current view of the world as a 2D list of
        terrain-type strings (plus "ship" and "goal" markers, and "unknown"
        for anything off the edge of the map).

        The returned grid is always oriented with "forward" at the top,
        "left" on the left, "right" on the right, and "behind" at the
        bottom - regardless of the ship's actual compass heading. This
        matches how a person on the ship would actually perceive their
        surroundings: relative to themselves, not relative to true north.
        """
        forward_row_vector, forward_col_vector = self.FORWARD_VECTOR_BY_HEADING[self.heading]
        right_row_vector, right_col_vector = self._get_right_vector_for_heading(self.heading)

        max_forward_distance = self.SIDEWAYS_VISIBILITY_RANGE + self.EXTRA_FORWARD_VISIBILITY_RANGE
        min_forward_distance = -self.SIDEWAYS_VISIBILITY_RANGE  # i.e. one tile behind
        max_sideways_distance = self.SIDEWAYS_VISIBILITY_RANGE

        view_grid = []
        # Iterate from furthest-ahead to furthest-behind so the first row
        # printed is what's farthest in front of the ship (i.e. "up" on the
        # page), matching how a person would expect a forward-facing view
        # to read from top to bottom.
        for forward_distance in range(max_forward_distance, min_forward_distance - 1, -1):
            view_row = []
            for sideways_distance in range(-max_sideways_distance, max_sideways_distance + 1):
                world_row = (
                    self.position[0]
                    + forward_distance * forward_row_vector
                    + sideways_distance * right_row_vector
                )
                world_col = (
                    self.position[1]
                    + forward_distance * forward_col_vector
                    + sideways_distance * right_col_vector
                )
                view_row.append(self._describe_tile(world_row, world_col, forward_distance, sideways_distance))
            view_grid.append(view_row)

        return view_grid

    def _describe_tile(self, world_row, world_col, forward_distance, sideways_distance):
        """What the ship sees at a given world position, described from the
        ship's own point of view."""
        if forward_distance == 0 and sideways_distance == 0:
            return "ship"

        map_size = self.archipelago_map.size
        if not (0 <= world_row < map_size and 0 <= world_col < map_size):
            return "unknown"

        if (world_row, world_col) == self.archipelago_map.ship_goal_position:
            return "goal"

        return self.archipelago_map.terrain_grid[world_row, world_col]

    def render_local_view(self, use_emoji=True):
        """Prints the ship's current egocentric view to the console."""
        if use_emoji:
            tile_symbols = {
                "water": "🟦",
                "beach": "🟨",
                "land": "🟩",
                "mountain": "🟫",
                "unknown": "⬛",
                "ship": "🟥",
                "goal": "🟪",
            }
        else:
            tile_symbols = {
                "water": "~",
                "beach": ".",
                "land": "#",
                "mountain": "^",
                "unknown": "?",
                "ship": "S",
                "goal": "G",
            }

        for view_row in self.get_local_view():
            print("".join(tile_symbols[tile] for tile in view_row))


if __name__ == "__main__":
    archipelago_map = ArchipelagoMap()
    print("Full map (ship position hidden):")
    archipelago_map.render()

    ship = Ship(archipelago_map)
    print(f"\nShip starts at {ship.position} facing {ship.heading}")
    print("Ship's local view:")
    ship.render_local_view()