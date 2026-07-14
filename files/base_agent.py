import re

import config
import crafter

# Constants
ACTION_NAMES = crafter.constants.actions
ACHIEVEMENTS = crafter.constants.achievements

SYSTEM_PROMPT = (
    "You are in a 2D survival game. Your objective is to survive as long as possible, "
    "while trying to complete as many achievements as you can.\n"
    "The available achievements are: " + ", ".join(ACHIEVEMENTS) + ".\n"
    "The available actions are: " + ", ".join(ACTION_NAMES) + ".\n"
    "You will be given your current local e"
)

# Longest names first so 'move_left' is matched before 'do', etc.
ACTION_PATTERN = re.compile(
    "|".join(rf"\b{re.escape(name)}\b" for name in sorted(ACTION_NAMES, key=len, reverse=True))
)


# ============================================================
# Base class
# ============================================================
class BaseAgent:
    """Turns observations into Crafter actions; generation is provider-specific."""

    provider = "base"

    def __init__(self, model_name):
        self.model_name = model_name
        self.history = []  # rolling [(action, outcome), ...]

    # -------------------- policy --------------------

    def choose_action(self, observation_text):
        """Return (action_name, raw_response, valid) for the observation."""
        response = self._generate(self._build_user_text(observation_text))
        action = self._extract_action(response)
        if action:
            return action, response, True
        return "noop", response, False

    def record_outcome(self, action_name, reward):
        """Append the last step to the rolling history shown in prompts."""
        outcome = f"reward {reward:+.1f}" if reward else "no reward"
        self.history.append((action_name, outcome))
        self.history = self.history[-config.HISTORY_LENGTH:]

    def reset_history(self):
        self.history = []

    def unload(self):
        """Free resources (only local models need this)."""

    # -------------------- internals --------------------

    def _generate(self, user_text):
        raise NotImplementedError

    def _build_user_text(self, observation_text):
        history = (
            "Recent actions:\n"
            + "\n".join(f"- {action} ({outcome})" for action, outcome in self.history)
            if self.history else "Recent actions: none yet"
        )
        return f"{observation_text}\n\n{history}\n\nNext action:"

    @staticmethod
    def _extract_action(response):
        """First match on the opening line (terse models answer immediately),
        otherwise the last match anywhere (chatty models conclude at the end)."""
        first_line = response.strip().split("\n", 1)[0]
        match = ACTION_PATTERN.search(first_line)
        if match:
            return match.group(0)
        matches = ACTION_PATTERN.findall(response)
        return matches[-1] if matches else None
