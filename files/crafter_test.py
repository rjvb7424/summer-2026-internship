import queue
import re
import threading
import time
from collections import deque

import crafter
import numpy as np

from viewer import CrafterViewer 


# Prompt and execution configuration.
ACTION_HISTORY_LIMIT = 12
SOLVER_POLL_SECONDS = 0.05
RESPONSE_PREVIEW_LENGTH = 240

# Symbols used in the text-only scene description.
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
OBJECT_SYMBOLS = {
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

CRAFTER_SYSTEM_PROMPT = """
You are an autonomous AI agent playing the Crafter survival environment.

PRIMARY OBJECTIVE
Unlock as many DIFFERENT achievements as possible before the episode ends.
A new achievement normally gives +1 reward. Repeating an achievement does not
increase the number of different achievements, so prioritize achievements that
are still incomplete while staying alive.

You must choose exactly one action per turn. Return only the action name.

VALID ACTIONS
noop, move_left, move_right, move_up, move_down, do, sleep,
place_stone, place_table, place_furnace, place_plant,
make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe,
make_wood_sword, make_stone_sword, make_iron_sword

WORLD AND CONTROL RULES
- The player is @ at the centre of the visible grid.
- The top of the grid is north/up.
- move_left, move_right, move_up, and move_down try to move one tile and also
  change the direction the player is facing.
- Only grass, path, and sand are normally walkable.
- Water, trees, stone, coal, iron, diamond, tables, furnaces, creatures, and
  plants block movement.
- Use do to interact with the tile directly in front of the player.
- If movement is blocked by a useful resource or creature, the movement still
  faces that target; use do next.
- Do not repeatedly alternate between opposite movements.
- Do not repeat an action that had NO EFFECT unless the state has changed.
- Explore deliberately. Prefer directions that have not just been tried.

ALL 22 ACHIEVEMENTS AND HOW TO UNLOCK THEM
1. collect_wood:
   Face a tree T and use do. No tool is required.

2. collect_sapling:
   Face grass . and use do. A sapling is received with a small probability, so
   this may require several attempts on grass.

3. collect_drink:
   Face water ~ and use do.

4. collect_stone:
   Have a wood_pickaxe, face stone #, and use do.

5. collect_coal:
   Have a wood_pickaxe, face coal c, and use do.

6. collect_iron:
   Have a stone_pickaxe, face iron i, and use do.

7. collect_diamond:
   Have an iron_pickaxe, face diamond d, and use do.

8. place_stone:
   Have at least 1 stone and face an empty valid placement tile. Use
   place_stone. It can bridge water or lava and can also be placed on grass,
   sand, or path.

9. place_table:
   Have at least 2 wood and face an empty grass, sand, or path tile. Use
   place_table.

10. place_furnace:
    Have at least 4 stone and face an empty grass, sand, or path tile. Use
    place_furnace.

11. place_plant:
    Have at least 1 sapling and face an empty grass tile. Use place_plant.

12. make_wood_pickaxe:
    Have at least 1 wood and stand next to a table. Use make_wood_pickaxe.

13. make_wood_sword:
    Have at least 1 wood and stand next to a table. Use make_wood_sword.

14. make_stone_pickaxe:
    Have at least 1 wood and 1 stone and stand next to a table. Use
    make_stone_pickaxe.

15. make_stone_sword:
    Have at least 1 wood and 1 stone and stand next to a table. Use
    make_stone_sword.

16. make_iron_pickaxe:
    Have at least 1 wood, 1 coal, and 1 iron and stand next to both a table and
    a furnace. Use make_iron_pickaxe.

17. make_iron_sword:
    Have at least 1 wood, 1 coal, and 1 iron and stand next to both a table and
    a furnace. Use make_iron_sword.

18. eat_cow:
    Face a cow C and use do repeatedly until it is defeated. A sword increases
    damage and makes this safer.

19. eat_plant:
    Face a mature plant P and use do. Newly placed plants take a long time to
    mature, but mature plants may also be found while exploring.

20. defeat_zombie:
    Face a zombie Z and use do repeatedly until defeated. Prefer using a sword
    and avoid fighting at low health.

21. defeat_skeleton:
    Face a skeleton S and use do repeatedly until defeated. Prefer using a
    sword and avoid fighting at low health.

22. wake_up:
    When energy is below its maximum, use sleep. Sleeping continues until energy
    is restored, then the player wakes and unlocks wake_up.

RECOMMENDED PROGRESSION
- First collect wood.
- Obtain at least 2 wood and place a table.
- Collect more wood and make a wood pickaxe and preferably a wood sword.
- Use the wood pickaxe to collect stone and coal.
- Place a furnace after collecting 4 stone.
- Make stone tools and collect iron.
- With a table and furnace nearby, make iron tools.
- Use the iron pickaxe to collect diamond.
- Pursue independent achievements whenever opportunities appear: drink water,
  collect a sapling, place a plant, place stone, fight creatures safely, eat a
  cow or mature plant, and sleep to wake up.

DECISION POLICY
1. Protect survival when health, food, drink, or energy is dangerously low.
2. Prefer an action that directly advances an incomplete achievement.
3. Use do when facing a useful resource or target.
4. If the previous movement was blocked, do not repeat it. Use do if the object
   ahead is useful; otherwise choose another direction.
5. If recent actions show left-right-left-right or up-down-up-down oscillation,
   break the loop by choosing a different action.
6. Never choose a crafting or placement action without its required resources
   and nearby structures.
7. Output exactly one valid action name and no explanation.
""".strip()


class CrafterTest:
    """Run one Crafter episode controlled by a prompt-based AI solver."""

    def __init__(
        self,
        max_steps=200,
        seed=0,
        show_simulation=True,
        record_directory=None,
        save_prompts=False,
    ):
        self.max_steps = max_steps
        self.seed = seed
        self.show_simulation = show_simulation
        self.record_directory = record_directory
        self.save_prompts = save_prompts

        self.env = crafter.Env(length=max_steps, seed=seed)
        if record_directory:
            self.env = crafter.Recorder(
                self.env,
                record_directory,
                save_stats=False,
                save_video=True,
                save_episode=False,
                video_size=(512, 512),
            )

    def run(self, solver, model="unknown", trial=1):
        """Run one episode and return a JSON-serializable result dictionary."""
        viewer = CrafterViewer() if self.show_simulation else None
        action_history = deque(maxlen=ACTION_HISTORY_LIMIT)
        trajectory = []

        total_reward = 0.0
        total_prompt_tokens = 0
        total_output_tokens = 0
        total_tokens = 0
        invalid_actions = 0
        solver_failed = False
        last_action = None
        last_action_result = "No previous action."
        last_reward = 0.0
        last_response = None
        done = False
        step = 0
        episode_start = time.time()

        self.env.reset()

        try:
            while not done and step < self.max_steps:
                messages = self.build_messages(
                    step=step,
                    action_history=action_history,
                    last_action=last_action,
                    last_action_result=last_action_result,
                    last_reward=last_reward,
                )

                solver_result = self._call_solver(
                    solver=solver,
                    messages=messages,
                    viewer=viewer,
                    model=model,
                    trial=trial,
                    step=step,
                    last_action=last_action,
                    last_reward=last_reward,
                    total_reward=total_reward,
                    last_response=last_response,
                )

                if not solver_result:
                    solver_failed = True
                    break

                raw_response = solver_result.get("text", "")
                action_name = self.extract_action(raw_response)
                if action_name is None:
                    action_name = "noop"
                    invalid_actions += 1

                position_before = self._player_position()
                inventory_before = self.env._player.inventory.copy()
                achievements_before = self.env._player.achievements.copy()
                facing_before = tuple(self.env._player.facing)

                action_index = self.env.action_names.index(action_name)
                _, reward, done, info = self.env.step(action_index)

                position_after = self._player_position()
                inventory_after = self.env._player.inventory.copy()
                achievements_after = self.env._player.achievements.copy()
                facing_after = tuple(self.env._player.facing)

                last_action_result = self._describe_action_result(
                    action=action_name,
                    position_before=position_before,
                    position_after=position_after,
                    facing_before=facing_before,
                    facing_after=facing_after,
                    inventory_before=inventory_before,
                    inventory_after=inventory_after,
                    achievements_before=achievements_before,
                    achievements_after=achievements_after,
                )

                step += 1
                last_action = action_name
                last_reward = float(reward)
                last_response = raw_response
                total_reward += float(reward)

                action_history.append({
                    "action": action_name,
                    "result": last_action_result,
                })

                total_prompt_tokens += solver_result.get("prompt_tokens") or 0
                total_output_tokens += solver_result.get("output_tokens") or 0
                total_tokens += solver_result.get("total_tokens") or 0

                decision = {
                    "step": step,
                    "action": action_name,
                    "action_result": last_action_result,
                    "reward": float(reward),
                    "position_before": list(position_before),
                    "position_after": list(position_after),
                    "raw_response": raw_response,
                    "elapsed_seconds": solver_result.get("elapsed_seconds"),
                    "prompt_tokens": solver_result.get("prompt_tokens"),
                    "output_tokens": solver_result.get("output_tokens"),
                    "thinking_tokens": solver_result.get("thinking_tokens"),
                    "total_tokens": solver_result.get("total_tokens"),
                    "is_partial": solver_result.get("is_partial", False),
                }
                if self.save_prompts:
                    decision["messages"] = messages
                trajectory.append(decision)

                if viewer:
                    viewer.render(
                        env=self.env,
                        model=model,
                        trial=trial,
                        step=step,
                        max_steps=self.max_steps,
                        status="Action completed",
                        last_action=last_action,
                        last_reward=last_reward,
                        total_reward=total_reward,
                        achievements=info["achievements"],
                        response_preview=self._response_preview(last_response),
                    )
        finally:
            if viewer:
                viewer.close()

        achievements = self.env._player.achievements.copy()
        inventory = self.env._player.inventory.copy()
        unlocked = [
            name
            for name, count in achievements.items()
            if count > 0
        ]

        return {
            "model_version": model,
            "trial": trial,
            "seed": self.seed,
            "max_steps": self.max_steps,
            "episode_steps": step,
            "total_reward": total_reward,
            "num_achievements": len(unlocked),
            "unlocked_achievements": unlocked,
            "achievements": achievements,
            "final_inventory": inventory,
            "invalid_actions": invalid_actions,
            "solver_failed": solver_failed,
            "elapsed_seconds": time.time() - episode_start,
            "prompt_tokens": total_prompt_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "trajectory": trajectory,
        }

    def build_messages(
        self,
        step,
        action_history,
        last_action,
        last_action_result,
        last_reward,
    ):
        """Build system instructions and the current user feedback message."""
        achievements = self.env._player.achievements
        completed = [
            name for name, count in achievements.items() if count > 0
        ]
        remaining = [
            name for name, count in achievements.items() if count == 0
        ]

        recent_actions = (
            "\n".join(
                f"- {item['action']}: {item['result']}"
                for item in action_history
            )
            if action_history
            else "- none"
        )

        loop_warning = self._loop_warning(action_history)

        user_feedback = "\n".join([
            "CURRENT STEP FEEDBACK",
            f"Step: {step}/{self.max_steps}",
            f"World position: {self._player_position()}",
            f"Facing: {FACING_NAMES.get(tuple(self.env._player.facing), 'unknown')}",
            f"Last action: {last_action or 'none'}",
            f"Last action result: {last_action_result}",
            f"Last reward: {last_reward:.2f}",
            f"Inventory: {self._format_inventory()}",
            f"Completed achievements ({len(completed)}/22): "
            f"{', '.join(completed) if completed else 'none'}",
            f"Remaining achievements: {', '.join(remaining)}",
            "",
            f"Loop status: {loop_warning}",
            "",
            "Adjacent tiles:",
            self._format_adjacent_tiles(),
            "",
            "Recent action outcomes:",
            recent_actions,
            "",
            "Visible local scene:",
            self._format_visible_scene(),
            "Legend: @ player, . grass, ~ water, # stone, : path, , sand, "
            "T tree, L lava, c coal, i iron, d diamond, b table, f furnace, "
            "C cow, Z zombie, S skeleton, A arrow, P plant, ? unknown.",
            "",
            "Choose the single action that best advances an incomplete "
            "achievement while avoiding loops and preserving survival.",
            "Return exactly one valid action name.",
        ])

        return [
            {
                "role": "system",
                "content": CRAFTER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_feedback,
            },
        ]

    def extract_action(self, response_text):
        """Extract one Crafter action name from the model response."""
        if not response_text:
            return None

        text = response_text.strip().lower()
        valid_actions = list(self.env.action_names)

        if text in valid_actions:
            return text

        explicit_patterns = [
            r'"action"\s*:\s*"([a-z_]+)"',
            r"action\s*[:=]\s*([a-z_]+)",
            r"final action\s*[:=]?\s*([a-z_]+)",
        ]
        for pattern in explicit_patterns:
            match = re.search(pattern, text)
            if match and match.group(1) in valid_actions:
                return match.group(1)

        matches = []
        for action in valid_actions:
            for match in re.finditer(rf"\b{re.escape(action)}\b", text):
                matches.append((match.start(), action))

        return max(matches)[1] if matches else None

    def _call_solver(
        self,
        solver,
        messages,
        viewer,
        model,
        trial,
        step,
        last_action,
        last_reward,
        total_reward,
        last_response,
    ):
        """Call the solver without freezing the Pygame event loop."""
        if viewer is None:
            return solver(messages)

        result_queue = queue.Queue(maxsize=1)

        def run_solver():
            try:
                result_queue.put(("result", solver(messages)))
            except Exception as error:
                result_queue.put(("error", error))

        thread = threading.Thread(target=run_solver, daemon=True)
        thread.start()
        started = time.time()

        while thread.is_alive():
            elapsed = time.time() - started
            viewer.render(
                env=self.env,
                model=model,
                trial=trial,
                step=step,
                max_steps=self.max_steps,
                status=f"Model thinking... {elapsed:.1f}s",
                last_action=last_action,
                last_reward=last_reward,
                total_reward=total_reward,
                achievements=self.env._player.achievements,
                response_preview=self._response_preview(last_response),
            )
            time.sleep(SOLVER_POLL_SECONDS)

        kind, payload = result_queue.get()
        if kind == "error":
            raise payload
        return payload

    def _describe_action_result(
        self,
        action,
        position_before,
        position_after,
        facing_before,
        facing_after,
        inventory_before,
        inventory_after,
        achievements_before,
        achievements_after,
    ):
        """Describe the observable effect of the previous action."""
        new_achievements = [
            name
            for name, count in achievements_after.items()
            if count > achievements_before.get(name, 0)
        ]
        inventory_changes = [
            f"{name} {inventory_before.get(name, 0)}->{amount}"
            for name, amount in inventory_after.items()
            if amount != inventory_before.get(name, 0)
        ]

        if new_achievements:
            return "SUCCESS; unlocked " + ", ".join(new_achievements)

        if inventory_changes:
            return "SUCCESS; inventory changed: " + ", ".join(
                inventory_changes
            )

        if action.startswith("move_"):
            if position_after != position_before:
                return (
                    f"SUCCESS; moved from {position_before} "
                    f"to {position_after}"
                )
            if facing_after != facing_before:
                return (
                    "BLOCKED; position did not change, but facing changed. "
                    "Do not repeat this movement."
                )
            return "BLOCKED; position and facing did not change."

        return (
            "NO EFFECT; the action did not change position, inventory, "
            "or achievements. Do not repeat it without a state change."
        )

    def _loop_warning(self, action_history):
        """Describe repeated or oscillating action patterns."""
        actions = [item["action"] for item in action_history]

        if len(actions) >= 4:
            last_four = actions[-4:]
            oscillations = {
                ("move_left", "move_right", "move_left", "move_right"),
                ("move_right", "move_left", "move_right", "move_left"),
                ("move_up", "move_down", "move_up", "move_down"),
                ("move_down", "move_up", "move_down", "move_up"),
            }
            if tuple(last_four) in oscillations:
                return (
                    "OSCILLATION DETECTED. Do not choose either of the last "
                    "two opposite movements. Select do or a perpendicular "
                    "direction."
                )

        if len(actions) >= 3 and len(set(actions[-3:])) == 1:
            return (
                f"REPETITION DETECTED: {actions[-1]} was selected three "
                "times. Choose a different action."
            )

        return "No movement loop detected."

    def _player_position(self):
        """Return the player's current world position."""
        return tuple(int(value) for value in self.env._player.pos)

    def _format_adjacent_tiles(self):
        """Describe the four tiles directly adjacent to the player."""
        world = self.env._world
        player = self.env._player
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

    def _format_visible_scene(self):
        """Describe the local map shown in the Crafter observation."""
        world = self.env._world
        player = self.env._player
        grid = np.array(self.env._local_view._grid)
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
                    symbol = OBJECT_SYMBOLS.get(
                        type(obj).__name__,
                        "O",
                    )
                else:
                    symbol = MATERIAL_SYMBOLS.get(material, "?")
                symbols.append(symbol)
            rows.append(" ".join(symbols))

        return "\n".join(rows)

    def _format_inventory(self):
        """Format current inventory and survival values."""
        return ", ".join(
            f"{name}={amount}"
            for name, amount in self.env._player.inventory.items()
        )

    def _response_preview(self, response):
        """Return a compact model-response preview."""
        if not response:
            return None
        compact = " ".join(response.split())
        return compact[:RESPONSE_PREVIEW_LENGTH]
