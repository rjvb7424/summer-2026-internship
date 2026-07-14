"""Crafter episode runner: text observation -> solver -> action, one trial per run()."""

import itertools
import re

import crafter

import config
from recorder import Recorder
from viewer import CrafterViewer

# Local view half-size: the model sees a (2R+1) x (2R+1) grid around the player.
VIEW_RADIUS = 4
HISTORY_LENGTH = 5
FACING_NAMES = {(-1, 0): "west", (1, 0): "east", (0, -1): "north", (0, 1): "south"}

PROMPT_TEMPLATE = """You are playing Crafter, a 2D survival game. Unlock achievements by \
collecting resources, crafting tools, and surviving.

Map around you ({size}x{size}, you are at the center, top row is north):
{grid}

You are facing {facing}. "do" interacts with the tile you are facing \
(chop tree, mine stone, attack, drink water, eat cow).
Status: {status}
Inventory: {inventory}
Achievements unlocked: {achievements}
Your last actions: {history}

Available actions:
{actions}

Rules: crafting needs a table nearby (place_table costs wood); iron tools also \
need a furnace; place_plant grows food; sleep restores energy.

Reply with exactly one action name from the list and nothing else."""


class CrafterTest:
    """Runs one Crafter trial where a language model chooses every action."""

    def __init__(self, max_steps, seed, show_simulation=False,
                 record_video=False, save_prompts=False):
        self.max_steps = max_steps
        self.seed = seed
        self.show_simulation = show_simulation
        self.record_video = record_video
        self.save_prompts = save_prompts

        self.env = crafter.Env(seed=seed)
        self.action_names = list(self.env.action_names)
        self.id_to_name = self._build_id_map()

    # ============================================================
    # Semantic map
    # ============================================================
    def _build_id_map(self):
        """Map semantic-view ids to readable names (materials and objects)."""
        id_to_name = {0: "empty"}
        pairs = itertools.chain(
            self.env._world._mat_ids.items(), self.env._sem_view._obj_ids.items()
        )
        for key, index in pairs:
            if key is None:
                continue
            name = key if isinstance(key, str) else key.__name__.lower()
            id_to_name[index] = str(name)
        return id_to_name

    def _local_grid(self, semantic, player_position):
        """Readable (2R+1)^2 grid around the player, top row = north."""
        x0, y0 = int(player_position[0]), int(player_position[1])
        width, height = semantic.shape
        rows = []
        for y in range(y0 - VIEW_RADIUS, y0 + VIEW_RADIUS + 1):
            cells = []
            for x in range(x0 - VIEW_RADIUS, x0 + VIEW_RADIUS + 1):
                if x == x0 and y == y0:
                    cells.append("YOU")
                elif 0 <= x < width and 0 <= y < height:
                    cells.append(self.id_to_name[int(semantic[x, y])])
                else:
                    cells.append("void")
            rows.append(" | ".join(cells))
        return "\n".join(rows)

    # ============================================================
    # Prompt and action parsing
    # ============================================================
    def _build_prompt(self, info, history):
        inventory = info["inventory"]
        status = ", ".join(
            f"{key}: {inventory[key]}" for key in ("health", "food", "drink", "energy")
        )
        items = ", ".join(
            f"{key}: {count}" for key, count in inventory.items()
            if count > 0 and key not in ("health", "food", "drink", "energy")
        ) or "nothing"
        unlocked = ", ".join(
            name for name, count in info["achievements"].items() if count > 0
        ) or "none"
        return PROMPT_TEMPLATE.format(
            size=2 * VIEW_RADIUS + 1,
            grid=self._local_grid(info["semantic"], info["player_pos"]),
            facing=FACING_NAMES[tuple(self.env._player.facing)],
            status=status,
            inventory=items,
            achievements=unlocked,
            history=", ".join(history) if history else "none",
            actions="\n".join(self.action_names),
        )

    def _parse_action(self, text):
        """Return the index of the last action name mentioned, or None."""
        pattern = "|".join(
            sorted((re.escape(name) for name in self.action_names), key=len, reverse=True)
        )
        matches = re.findall(rf"\b(?:{pattern})\b", text.lower())
        return self.action_names.index(matches[-1]) if matches else None

    # ============================================================
    # Episode loop
    # ============================================================
    def run(self, solver, model, trial):
        """Run one full episode and return the trial record."""
        self.env.reset()
        _, _, _, info = self.env.step(self.action_names.index("noop"))

        viewer = CrafterViewer() if self.show_simulation else None
        recorder = Recorder(model, trial) if self.record_video else None

        total_reward = 0.0
        invalid_actions = 0
        steps = 0
        stopped_by_user = False
        history = []
        transcript = []
        last_action = None
        response_preview = None

        try:
            for step in range(1, self.max_steps + 1):
                prompt = self._build_prompt(info, history[-HISTORY_LENGTH:])

                if viewer and not viewer.render(
                    self.env, model, trial, step, self.max_steps, "Waiting for model...",
                    last_action, 0.0, total_reward, info["achievements"], response_preview,
                ):
                    stopped_by_user = True
                    break

                response = solver(prompt)
                text = response["text"] if response else ""
                action_index = self._parse_action(text)
                if action_index is None:
                    invalid_actions += 1
                    action_index = self.action_names.index("noop")
                last_action = self.action_names[action_index]
                response_preview = text.strip()[-300:]

                if self.save_prompts:
                    transcript.append({"step": step, "prompt": prompt, "response": text})

                _, reward, done, info = self.env.step(action_index)
                total_reward += reward
                steps = step

                if recorder:
                    recorder.add_frame(self.env.render((512, 512)))
                if viewer and not viewer.render(
                    self.env, model, trial, step, self.max_steps, "Acting",
                    last_action, reward, total_reward, info["achievements"], response_preview,
                ):
                    stopped_by_user = True
                    break

                history.append(last_action)
                if done:
                    break
        finally:
            if recorder:
                recorder.close()
            if viewer:
                viewer.close()

        record = {
            "model_version": model,
            "trial": trial,
            "seed": self.seed,
            "steps": max(steps, 1),
            "total_reward": total_reward,
            "invalid_actions": invalid_actions,
            "achievements": info["achievements"],
            "achievements_unlocked": sum(
                1 for count in info["achievements"].values() if count > 0
            ),
            "stopped_by_user": stopped_by_user,
        }
        if self.save_prompts:
            record["transcript"] = transcript
        return record