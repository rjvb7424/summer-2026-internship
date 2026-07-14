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
