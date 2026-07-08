import queue
import re
import threading
import time
from collections import Counter, deque

import crafter
import numpy as np

from viewer import CrafterViewer


ACTION_HISTORY_LIMIT = 12
SOLVER_POLL_SECONDS = 0.05
RESPONSE_PREVIEW_LENGTH = 240
MAX_CANDIDATE_ACTIONS = 4

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
OPPOSITE_ACTIONS = {
    "move_left": "move_right",
    "move_right": "move_left",
    "move_up": "move_down",
    "move_down": "move_up",
}

CRAFTER_SYSTEM_PROMPT = """
You are an autonomous agent playing Crafter.

PRIMARY GOAL
Unlock as many of the 22 different achievements as possible before the episode
ends. New achievements are more valuable than repeating completed ones. Stay
alive, follow the progression chain, and avoid movement loops.

CONTROL RULES
- The player is @ at the centre of the local map.
- Movement changes facing even when the destination is blocked.
- Use do on the tile directly in front of the player.
- If a movement faces a useful blocked target, use do on the next turn.
- You may choose only from ALLOWED ACTIONS FOR THIS STEP.
- Never alternate left-right-left-right or up-down-up-down.
- Return exactly one action name and no explanation.

ACHIEVEMENTS
collect_wood: face tree T and use do.
collect_sapling: face grass . and use do; success probability is 10 percent.
collect_drink: face water ~ and use do.
collect_stone: have a wood pickaxe, face stone #, use do.
collect_coal: have a wood pickaxe, face coal c, use do.
collect_iron: have a stone pickaxe, face iron i, use do.
collect_diamond: have an iron pickaxe, face diamond d, use do.
place_stone: have 1 stone, face a valid empty tile, use place_stone.
place_table: have 2 wood, face empty grass, sand, or path, use place_table.
place_furnace: have 4 stone, face empty grass, sand, or path, use place_furnace.
place_plant: have 1 sapling, face empty grass, use place_plant.
make_wood_pickaxe: have 1 wood and stand next to a table.
make_wood_sword: have 1 wood and stand next to a table.
make_stone_pickaxe: have 1 wood and 1 stone and stand next to a table.
make_stone_sword: have 1 wood and 1 stone and stand next to a table.
make_iron_pickaxe: have 1 wood, 1 coal, and 1 iron and stand next to a table
and furnace.
make_iron_sword: have 1 wood, 1 coal, and 1 iron and stand next to a table
and furnace.
eat_cow: face cow C and use do until defeated.
eat_plant: face a mature plant P and use do.
defeat_zombie: face zombie Z and use do until defeated.
defeat_skeleton: face skeleton S and use do until defeated.
wake_up: use sleep when energy is below maximum and wake after recovery.

MAIN PROGRESSION
Collect wood, place a table, make wood tools, collect stone and coal, place a
furnace, make stone tools, collect iron, make iron tools, then collect diamond.
Take easy independent achievements whenever they are directly available.
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
        position_visits = Counter()
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
        position_visits[self._player_position()] += 1

        try:
            while not done and step < self.max_steps:
                current_goal = self._select_current_goal()
                candidate_actions = self._candidate_actions(
                    current_goal=current_goal,
                    action_history=action_history,
                    position_visits=position_visits,
                )
                messages = self.build_messages(
                    step=step,
                    action_history=action_history,
                    last_action=last_action,
                    last_action_result=last_action_result,
                    last_reward=last_reward,
                    current_goal=current_goal,
                    candidate_actions=candidate_actions,
                )

                solver_result = self._call_solver(
                    solver=solver,
                    messages=messages,
                    candidate_actions=candidate_actions,
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
                if action_name not in candidate_actions:
                    invalid_actions += 1
                    action_name = candidate_actions[0]

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
                position_visits[position_after] += 1
                action_history.append({
                    "action": action_name,
                    "result": last_action_result,
                })

                total_prompt_tokens += solver_result.get("prompt_tokens") or 0
                total_output_tokens += solver_result.get("output_tokens") or 0
                total_tokens += solver_result.get("total_tokens") or 0

                decision = {
                    "step": step,
                    "current_goal": current_goal,
                    "candidate_actions": candidate_actions,
                    "candidate_scores": solver_result.get("candidate_scores"),
                    "decision_mode": solver_result.get("decision_mode"),
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
                        status=f"Goal: {current_goal}",
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
            name for name, count in achievements.items() if count > 0
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
        current_goal,
        candidate_actions,
    ):
        """Build permanent instructions and compact current-state feedback."""
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

        user_feedback = "\n".join([
            "CURRENT STATE",
            f"Step: {step}/{self.max_steps}",
            f"Strategic goal: {current_goal}",
            f"Position: {self._player_position()}",
            f"Facing: {FACING_NAMES.get(tuple(self.env._player.facing), 'unknown')}",
            f"Last action: {last_action or 'none'}",
            f"Last action result: {last_action_result}",
            f"Last reward: {last_reward:.2f}",
            f"Inventory: {self._format_inventory()}",
            f"Completed achievements ({len(completed)}/22): "
            f"{', '.join(completed) if completed else 'none'}",
            f"Remaining achievements: {', '.join(remaining)}",
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
            "ALLOWED ACTIONS FOR THIS STEP: " + ", ".join(candidate_actions),
            "Choose the action that best advances the strategic goal.",
            "Return exactly one allowed action name.",
        ])

        return [
            {"role": "system", "content": CRAFTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_feedback},
        ]

    def extract_action(self, response_text):
        """Extract one Crafter action name from the model response."""
        if not response_text:
            return None

        text = response_text.strip().lower()
        valid_actions = list(self.env.action_names)
        if text in valid_actions:
            return text

        matches = []
        for action in valid_actions:
            for match in re.finditer(rf"\b{re.escape(action)}\b", text):
                matches.append((match.start(), action))
        return max(matches)[1] if matches else None

    def _call_solver(
        self,
        solver,
        messages,
        candidate_actions,
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
        def invoke_solver():
            return solver(
                messages,
                candidate_actions=candidate_actions,
            )

        if viewer is None:
            return invoke_solver()

        result_queue = queue.Queue(maxsize=1)

        def run_solver():
            try:
                result_queue.put(("result", invoke_solver()))
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
                status=f"Model choosing action... {elapsed:.1f}s",
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

    def _candidate_actions(
        self,
        current_goal,
        action_history,
        position_visits,
    ):
        """Build a small strategic action mask for the current state."""
        survival_action = self._urgent_survival_action()
        if survival_action:
            return [survival_action]

        interaction_action = self._front_interaction_action(current_goal)
        if interaction_action:
            return [interaction_action]

        craft_actions = self._feasible_craft_actions()
        if craft_actions:
            return craft_actions[:MAX_CANDIDATE_ACTIONS]

        place_actions = self._feasible_place_actions()
        if place_actions:
            return place_actions[:MAX_CANDIDATE_ACTIONS]

        target_directions = self._adjacent_target_directions(current_goal)
        target_directions = self._remove_looping_actions(
            target_directions,
            action_history,
        )
        if target_directions:
            return target_directions[:MAX_CANDIDATE_ACTIONS]

        exploration_actions = self._exploration_actions(position_visits)
        exploration_actions = self._remove_looping_actions(
            exploration_actions,
            action_history,
        )
        if exploration_actions:
            return exploration_actions[:MAX_CANDIDATE_ACTIONS]

        return ["noop"]

    def _urgent_survival_action(self):
        """Return a forced survival action when one is immediately required."""
        inventory = self.env._player.inventory
        achievements = self.env._player.achievements
        if inventory["energy"] <= 2 and achievements["wake_up"] == 0:
            return "sleep"
        return None

    def _front_interaction_action(self, current_goal):
        """Force do when the player is already facing a useful target."""
        material, obj = self._front_tile()
        inventory = self.env._player.inventory
        achievements = self.env._player.achievements

        if obj is not None:
            object_name = type(obj).__name__
            if object_name == "Cow" and achievements["eat_cow"] == 0:
                return "do"
            if (
                object_name == "Zombie"
                and achievements["defeat_zombie"] == 0
                and inventory["health"] >= 5
            ):
                return "do"
            if (
                object_name == "Skeleton"
                and achievements["defeat_skeleton"] == 0
                and inventory["health"] >= 5
            ):
                return "do"
            if (
                object_name == "Plant"
                and achievements["eat_plant"] == 0
                and getattr(obj, "ripe", False)
            ):
                return "do"
            return None

        if material == "tree" and self._needs_wood():
            return "do"
        if material == "water" and (
            achievements["collect_drink"] == 0
            or inventory["drink"] < 8
        ):
            return "do"
        if material == "grass" and current_goal == "collect_sapling":
            return "do"
        if (
            material == "stone"
            and inventory["wood_pickaxe"] > 0
            and self._needs_stone()
        ):
            return "do"
        if (
            material == "coal"
            and inventory["wood_pickaxe"] > 0
            and self._needs_coal()
        ):
            return "do"
        if (
            material == "iron"
            and inventory["stone_pickaxe"] > 0
            and self._needs_iron()
        ):
            return "do"
        if (
            material == "diamond"
            and inventory["iron_pickaxe"] > 0
            and achievements["collect_diamond"] == 0
        ):
            return "do"
        return None

    def _feasible_craft_actions(self):
        """Return incomplete crafting achievements whose requirements are met."""
        inventory = self.env._player.inventory
        achievements = self.env._player.achievements
        nearby = self._nearby_materials()
        has_table = "table" in nearby
        has_furnace = "furnace" in nearby
        actions = []

        if has_table and inventory["wood"] >= 1:
            if achievements["make_wood_pickaxe"] == 0:
                actions.append("make_wood_pickaxe")
            if achievements["make_wood_sword"] == 0:
                actions.append("make_wood_sword")

        if has_table and inventory["wood"] >= 1 and inventory["stone"] >= 1:
            if achievements["make_stone_pickaxe"] == 0:
                actions.append("make_stone_pickaxe")
            if achievements["make_stone_sword"] == 0:
                actions.append("make_stone_sword")

        if (
            has_table
            and has_furnace
            and inventory["wood"] >= 1
            and inventory["coal"] >= 1
            and inventory["iron"] >= 1
        ):
            if achievements["make_iron_pickaxe"] == 0:
                actions.append("make_iron_pickaxe")
            if achievements["make_iron_sword"] == 0:
                actions.append("make_iron_sword")

        return actions

    def _feasible_place_actions(self):
        """Return incomplete placement achievements that can work now."""
        inventory = self.env._player.inventory
        achievements = self.env._player.achievements
        material, obj = self._front_tile()
        if obj is not None:
            return []

        actions = []
        if (
            achievements["place_table"] == 0
            and inventory["wood"] >= 2
            and material in {"grass", "sand", "path"}
        ):
            actions.append("place_table")
        if (
            achievements["place_furnace"] == 0
            and inventory["stone"] >= 4
            and material in {"grass", "sand", "path"}
        ):
            actions.append("place_furnace")
        if (
            achievements["place_stone"] == 0
            and inventory["stone"] >= 1
            and material in {"grass", "sand", "path", "water", "lava"}
        ):
            actions.append("place_stone")
        if (
            achievements["place_plant"] == 0
            and inventory["sapling"] >= 1
            and material == "grass"
        ):
            actions.append("place_plant")
        return actions

    def _adjacent_target_directions(self, current_goal):
        """Rank directions that face resources or achievement targets."""
        world = self.env._world
        player = self.env._player
        scored = []

        for action, direction in DIRECTION_VECTORS.items():
            material, obj = world[player.pos + np.array(direction)]
            score = self._target_score(
                material=material,
                obj=obj,
                current_goal=current_goal,
            )
            if score > 0:
                scored.append((score, action))

        scored.sort(reverse=True)
        return [action for _, action in scored]

    def _target_score(self, material, obj, current_goal):
        """Return the strategic value of facing one adjacent target."""
        inventory = self.env._player.inventory
        achievements = self.env._player.achievements

        if obj is not None:
            object_name = type(obj).__name__
            if object_name == "Cow" and achievements["eat_cow"] == 0:
                return 85
            if (
                object_name == "Plant"
                and achievements["eat_plant"] == 0
                and getattr(obj, "ripe", False)
            ):
                return 85
            if (
                object_name == "Zombie"
                and achievements["defeat_zombie"] == 0
                and inventory["health"] >= 5
            ):
                return 55
            if (
                object_name == "Skeleton"
                and achievements["defeat_skeleton"] == 0
                and inventory["health"] >= 5
            ):
                return 50
            return 0

        goal_materials = {
            "collect_wood": "tree",
            "collect_sapling": "grass",
            "collect_drink": "water",
            "collect_stone": "stone",
            "collect_coal": "coal",
            "collect_iron": "iron",
            "collect_diamond": "diamond",
        }
        if goal_materials.get(current_goal) == material:
            return 100

        if material == "tree" and self._needs_wood():
            return 90
        if material == "water" and (
            achievements["collect_drink"] == 0
            or inventory["drink"] < 6
        ):
            return 80
        if (
            material == "stone"
            and inventory["wood_pickaxe"] > 0
            and self._needs_stone()
        ):
            return 75
        if (
            material == "coal"
            and inventory["wood_pickaxe"] > 0
            and self._needs_coal()
        ):
            return 70
        if (
            material == "iron"
            and inventory["stone_pickaxe"] > 0
            and self._needs_iron()
        ):
            return 70
        if (
            material == "diamond"
            and inventory["iron_pickaxe"] > 0
            and achievements["collect_diamond"] == 0
        ):
            return 95
        return 0

    def _exploration_actions(self, position_visits):
        """Prefer walkable directions leading to less visited positions."""
        world = self.env._world
        player = self.env._player
        scored = []

        for action, direction in DIRECTION_VECTORS.items():
            target = player.pos + np.array(direction)
            material, obj = world[target]
            if obj is not None or material not in player.walkable:
                continue

            target_tuple = tuple(int(value) for value in target)
            score = 20 - 5 * position_visits[target_tuple]
            scored.append((score, action))

        scored.sort(reverse=True)
        return [action for _, action in scored]

    def _remove_looping_actions(self, actions, action_history):
        """Remove actions known to continue a repeated or oscillating loop."""
        if not actions:
            return []

        forbidden = set()
        recent_actions = [item["action"] for item in action_history]

        if action_history:
            last_item = action_history[-1]
            if (
                last_item["action"].startswith("move_")
                and last_item["result"].startswith("BLOCKED")
            ):
                forbidden.add(last_item["action"])

        if len(recent_actions) >= 4:
            a, b, c, d = recent_actions[-4:]
            if a == c and b == d and OPPOSITE_ACTIONS.get(a) == b:
                forbidden.update({a, b})

        filtered = [action for action in actions if action not in forbidden]
        return filtered or actions

    def _select_current_goal(self):
        """Select the next strategic achievement or resource objective."""
        inventory = self.env._player.inventory
        achievements = self.env._player.achievements
        nearby = self._nearby_materials()

        if achievements["collect_drink"] == 0 and inventory["drink"] <= 5:
            return "collect_drink"
        if achievements["collect_wood"] == 0 or self._needs_wood():
            return "collect_wood"
        if achievements["place_table"] == 0:
            return "place_table"
        if (
            achievements["make_wood_pickaxe"] == 0
            and "table" in nearby
            and inventory["wood"] >= 1
        ):
            return "make_wood_pickaxe"
        if achievements["make_wood_pickaxe"] == 0:
            return "return_to_table_and_make_wood_pickaxe"
        if self._needs_stone():
            return "collect_stone"
        if self._needs_coal():
            return "collect_coal"
        if achievements["place_furnace"] == 0:
            return "place_furnace"
        if achievements["make_stone_pickaxe"] == 0:
            return "make_stone_pickaxe"
        if self._needs_iron():
            return "collect_iron"
        if achievements["make_iron_pickaxe"] == 0:
            return "make_iron_pickaxe"
        if achievements["collect_diamond"] == 0:
            return "collect_diamond"
        if achievements["collect_sapling"] == 0:
            return "collect_sapling"

        remaining = [
            name for name, count in achievements.items() if count == 0
        ]
        return remaining[0] if remaining else "survive"

    def _needs_wood(self):
        inventory = self.env._player.inventory
        achievements = self.env._player.achievements
        remaining_wood_achievements = any(
            achievements[name] == 0
            for name in (
                "place_table",
                "make_wood_pickaxe",
                "make_wood_sword",
                "make_stone_pickaxe",
                "make_stone_sword",
                "make_iron_pickaxe",
                "make_iron_sword",
            )
        )
        return achievements["collect_wood"] == 0 or (
            remaining_wood_achievements and inventory["wood"] < 4
        )

    def _needs_stone(self):
        inventory = self.env._player.inventory
        achievements = self.env._player.achievements
        remaining_stone_achievements = any(
            achievements[name] == 0
            for name in (
                "place_stone",
                "place_furnace",
                "make_stone_pickaxe",
                "make_stone_sword",
            )
        )
        return achievements["collect_stone"] == 0 or (
            remaining_stone_achievements and inventory["stone"] < 7
        )

    def _needs_coal(self):
        inventory = self.env._player.inventory
        achievements = self.env._player.achievements
        return achievements["collect_coal"] == 0 or (
            (
                achievements["make_iron_pickaxe"] == 0
                or achievements["make_iron_sword"] == 0
            )
            and inventory["coal"] < 2
        )

    def _needs_iron(self):
        inventory = self.env._player.inventory
        achievements = self.env._player.achievements
        return achievements["collect_iron"] == 0 or (
            (
                achievements["make_iron_pickaxe"] == 0
                or achievements["make_iron_sword"] == 0
            )
            and inventory["iron"] < 2
        )

    def _front_tile(self):
        player = self.env._player
        target = player.pos + np.array(player.facing)
        return self.env._world[target]

    def _nearby_materials(self):
        """Return materials in the 3x3 neighborhood around the player."""
        materials = set()
        player_position = self.env._player.pos
        world = self.env._world

        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                position = player_position + np.array((offset_x, offset_y))
                material, _ = world[position]
                materials.add(material)
        return materials

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
                    "BLOCKED; position did not change, but facing changed."
                )
            return "BLOCKED; position and facing did not change."
        return "NO EFFECT; state did not change."

    def _player_position(self):
        return tuple(int(value) for value in self.env._player.pos)

    def _format_adjacent_tiles(self):
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
                    "walkable" if material in player.walkable else "blocked"
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
                    symbol = OBJECT_SYMBOLS.get(type(obj).__name__, "O")
                else:
                    symbol = MATERIAL_SYMBOLS.get(material, "?")
                symbols.append(symbol)
            rows.append(" ".join(symbols))
        return "\n".join(rows)

    def _format_inventory(self):
        return ", ".join(
            f"{name}={amount}"
            for name, amount in self.env._player.inventory.items()
        )

    def _response_preview(self, response):
        if not response:
            return None
        compact = " ".join(response.split())
        return compact[:RESPONSE_PREVIEW_LENGTH]
