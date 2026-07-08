import queue
import re
import threading
import time
from collections import deque

import crafter
import numpy as np

from viewer import CrafterViewer

# Prompt and execution configuration
ACTION_HISTORY_LIMIT = 8
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


class UserClosedViewer(Exception):
    """Raised when the user closes the live simulation window."""


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
                save_stats=True,
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
        stopped_by_user = False
        solver_failed = False
        last_action = None
        last_reward = 0.0
        last_response = None
        done = False
        step = 0
        episode_start = time.time()

        self.env.reset()

        try:
            while not done and step < self.max_steps:
                prompt = self.build_prompt(
                    step=step,
                    action_history=action_history,
                    last_reward=last_reward,
                )

                solver_result = self._call_solver(
                    solver=solver,
                    prompt=prompt,
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

                action_index = self.env.action_names.index(action_name)
                _, reward, done, info = self.env.step(action_index)

                step += 1
                last_action = action_name
                last_reward = float(reward)
                last_response = raw_response
                total_reward += float(reward)
                action_history.append(action_name)

                total_prompt_tokens += solver_result.get("prompt_tokens") or 0
                total_output_tokens += solver_result.get("output_tokens") or 0
                total_tokens += solver_result.get("total_tokens") or 0

                decision = {
                    "step": step,
                    "action": action_name,
                    "reward": float(reward),
                    "raw_response": raw_response,
                    "elapsed_seconds": solver_result.get("elapsed_seconds"),
                    "prompt_tokens": solver_result.get("prompt_tokens"),
                    "output_tokens": solver_result.get("output_tokens"),
                    "thinking_tokens": solver_result.get("thinking_tokens"),
                    "total_tokens": solver_result.get("total_tokens"),
                    "is_partial": solver_result.get("is_partial", False),
                }
                if self.save_prompts:
                    decision["prompt"] = prompt
                trajectory.append(decision)

                if viewer and not viewer.render(
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
                ):
                    raise UserClosedViewer()

        except UserClosedViewer:
            stopped_by_user = True
        finally:
            if viewer:
                viewer.close()

        achievements = self.env._player.achievements.copy()
        inventory = self.env._player.inventory.copy()
        unlocked = [name for name, count in achievements.items() if count > 0]

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
            "stopped_by_user": stopped_by_user,
            "elapsed_seconds": time.time() - episode_start,
            "prompt_tokens": total_prompt_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "trajectory": trajectory,
        }

    def build_prompt(self, step, action_history, last_reward):
        """Build a text-only observation prompt for any chat-style model."""
        actions = ", ".join(self.env.action_names)
        recent_actions = ", ".join(action_history) if action_history else "none"
        inventory = self._format_inventory()
        achievements = self._format_achievements()
        scene = self._format_visible_scene()
        facing = FACING_NAMES.get(tuple(self.env._player.facing), "unknown")

        return "\n".join([
            "You control the player in the Crafter survival environment.",
            "Your objective is to survive and unlock as many different achievements as possible.",
            "Choose exactly one valid action for the current step.",
            "",
            f"Step: {step}/{self.max_steps}",
            f"Facing: {facing}",
            f"Last reward: {last_reward:.2f}",
            f"Recent actions: {recent_actions}",
            f"Inventory: {inventory}",
            f"Achievements already unlocked: {achievements}",
            "",
            "Visible scene:",
            scene,
            "Legend: @ player, . grass, ~ water, # stone, : path, , sand, "
            "T tree, L lava, c coal, i iron, d diamond, b table, f furnace, "
            "C cow, Z zombie, S skeleton, A arrow, P plant, ? outside/unknown.",
            "The centre of the grid is the player. The top row is north/up.",
            "The action 'do' interacts with the tile directly in front of the player.",
            "Movement also changes the direction the player is facing.",
            "",
            f"Valid actions: {actions}",
            "Respond with only one action name and no explanation.",
        ])

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
        prompt,
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
            return solver(prompt)

        result_queue = queue.Queue(maxsize=1)

        def run_solver():
            try:
                result_queue.put(("result", solver(prompt)))
            except Exception as error:
                result_queue.put(("error", error))

        thread = threading.Thread(target=run_solver, daemon=True)
        thread.start()
        started = time.time()

        while thread.is_alive():
            elapsed = time.time() - started
            status = f"Model thinking... {elapsed:.1f}s"
            if not viewer.render(
                env=self.env,
                model=model,
                trial=trial,
                step=step,
                max_steps=self.max_steps,
                status=status,
                last_action=last_action,
                last_reward=last_reward,
                total_reward=total_reward,
                achievements=self.env._player.achievements,
                response_preview=self._response_preview(last_response),
            ):
                raise UserClosedViewer()
            time.sleep(SOLVER_POLL_SECONDS)

        kind, payload = result_queue.get()
        if kind == "error":
            raise payload
        return payload

    def _format_visible_scene(self):
        """Describe the same local map shown in the Crafter observation."""
        world = self.env._world
        player = self.env._player
        grid = np.array(self.env._local_view._grid)
        offset = grid // 2
        rows = []

        for local_y in range(grid[1]):
            symbols = []
            for local_x in range(grid[0]):
                position = player.pos + np.array((local_x, local_y)) - offset
                material, obj = world[position]

                if obj is not None:
                    symbol = OBJECT_SYMBOLS.get(type(obj).__name__, "O")
                else:
                    symbol = MATERIAL_SYMBOLS.get(material, "?")
                symbols.append(symbol)
            rows.append(" ".join(symbols))

        return "\n".join(rows)

    def _format_inventory(self):
        inventory = self.env._player.inventory
        values = [f"{name}={amount}" for name, amount in inventory.items()]
        return ", ".join(values)

    def _format_achievements(self):
        achievements = [
            name
            for name, count in self.env._player.achievements.items()
            if count > 0
        ]
        return ", ".join(achievements) if achievements else "none"

    def _response_preview(self, response):
        if not response:
            return None
        compact = " ".join(response.split())
        return compact[:RESPONSE_PREVIEW_LENGTH]
