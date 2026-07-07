"""
Interactive manual playtest for the ship navigation experiment.

Generates a fresh archipelago, shows you the full map (with the goal marked
but the ship hidden - exactly what the AI solver would see), then lets you
steer the ship with egocentric commands until you reach the goal.

Run with:
    python3 perspective_test.py
"""

from archipelago import ArchipelagoMap
from ship import Ship


COMMAND_ALIASES = {
    "f": "forward",
    "forward": "forward",
    "b": "backward",
    "backward": "backward",
    "l": "left",
    "left": "left",
    "r": "right",
    "right": "right",
    "m": "map",
    "map": "map",
    "q": "quit",
    "quit": "quit",
    "n": "face_north",
    "north": "face_north",
    "s": "face_south",
    "south": "face_south",
    "e": "face_east",
    "east": "face_east",
    "w": "face_west",
    "west": "face_west",
}

HELP_TEXT = (
    "Commands: [f]orward  [b]ackward  [l]eft (turn)  [r]ight (turn)  "
    "[n/s/e/w] face that compass direction directly  "
    "[m]ap (reshow full map)  [q]uit"
)


def apply_command(ship, command):
    """Executes one command against the ship. Returns True if the game
    should keep going, False if the player asked to quit."""
    if command == "forward":
        if not ship.move_forward():
            print("Blocked - can't sail that way (land, beach, or map edge).")
    elif command == "backward":
        if not ship.move_backward():
            print("Blocked - can't sail that way (land, beach, or map edge).")
    elif command == "left":
        ship.turn_left()
        print(f"Turned left, now facing {ship.heading}.")
    elif command == "right":
        ship.turn_right()
        print(f"Turned right, now facing {ship.heading}.")
    elif command == "map":
        print("\nFull map (ship position hidden):")
        ship.archipelago_map.render()
    elif command in ("face_north", "face_south", "face_east", "face_west"):
        heading = command.removeprefix("face_")
        ship.face(heading)
        print(f"Now facing {ship.heading}.")
    elif command == "quit":
        return False
    return True


def run_perspective_test(size=10):
    archipelago_map = ArchipelagoMap(size=size)
    ship = Ship(archipelago_map)

    print(f"New map generated (seed={archipelago_map.seed}).")
    print("Full map (ship position hidden):")
    archipelago_map.render()
    print(f"\nGoal is marked 🟪. {HELP_TEXT}\n")

    turn_number = 0
    while True:
        if ship.position == archipelago_map.ship_goal_position:
            print(f"\n🎉 Reached the goal in {turn_number} moves!")
            return

        turn_number += 1
        print(f"\n--- Turn {turn_number} (facing {ship.heading}) ---")
        ship.render_local_view()

        raw_input_text = input("\nYour move: ").strip().lower()
        command = COMMAND_ALIASES.get(raw_input_text)
        if command is None:
            print(f"Unrecognized command '{raw_input_text}'. {HELP_TEXT}")
            turn_number -= 1  # an invalid command doesn't count as a real turn
            continue

        should_continue = apply_command(ship, command)
        if not should_continue:
            print(f"\nQuit after {turn_number - 1} moves. "
                  f"Final position was not revealed (as intended).")
            return


if __name__ == "__main__":
    run_perspective_test()
