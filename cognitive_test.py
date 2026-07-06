import random
import copy
import re

class Paper:
    """Class representing a piece of paper that can be folded and punched."""
    def __init__(self, current_width=16, current_height=16, layer=1):
        # Store the original dimensions of the paper
        self.ORIGINAL_WIDTH = current_width
        self.ORIGINAL_HEIGHT = current_height
        # Store the current dimensions of the paper
        self.current_width = current_width
        self.current_height = current_height
        # The current face of the paper, represented as a 2D grid
        self.face = self.generate_face()
        # Track the orientation of folds applied to the paper
        self.fold_history = []
        # Track the number of layers of paper (doubles with each fold)
        self.layer = layer

    def generate_face(self):
        """Generate a blank 2D grid matching the current dimensions."""
        return [[0 for _ in range(self.current_width)] for _ in range(self.current_height)]

    def face_to_string(self):
        """Render the current face as a plain string grid."""
        return "\n".join(" ".join(str(c) for c in row) for row in self.face)

    def fold(self, orientation="north"):
        """Fold the paper in the given orientation."""
        if orientation not in ("north", "south", "east", "west"):
            raise ValueError(f"Unknown fold orientation: {orientation!r}")
        # Halve the current dimensions based on the fold orientation
        if orientation in ("north", "south"):
            self.current_height //= 2
        else:
            self.current_width //= 2
        # Double the layer count to reflect the fold
        self.layer *= 2
        self.fold_history.append(orientation)
        self.face = self.generate_face()

    def punch(self, x, y):
        """Punch a hole at the given (x, y) coordinates on the current face of the paper.
        Returns True if the punch was successful (in bounds), False otherwise."""
        if 0 <= x < self.current_width and 0 <= y < self.current_height:
            self.face[y][x] = 1
            return True
        return False

    def unfold(self):
        """Unfold the paper back to its original state, poping the fold_history in the process."""
        while self.layer != 1:
            last_fold = self.fold_history.pop()
            if last_fold == "north":
                # Current face was the north half; mirror it downward.
                new_face = self.face + self.face[::-1]
            elif last_fold == "south":
                # Current face was the south half; mirror it upward.
                new_face = self.face[::-1] + self.face
            elif last_fold == "west":
                # Current face was the west half; mirror it rightward.
                new_face = [row + row[::-1] for row in self.face]
            elif last_fold == "east":
                # Current face was the east half; mirror it leftward.
                new_face = [row[::-1] + row for row in self.face]

            if last_fold in ("north", "south"):
                self.current_height *= 2
            else:
                self.current_width *= 2
            # Update the face and layer count after unfolding
            self.face = new_face
            self.layer //= 2

class CognitiveTest:
    """Class representing a cognitive test involving folding and punching paper."""
    def __init__(self, width=16, height=16):
        # Initialize the test with a Paper instance
        self.test_paper = Paper(width, height)
        self.choices = {"A": None, "B": None, "C": None, "D": None, "E": None}
        # Track the sequence of fold orientations applied to the test paper
        # This is done seperately because paper.unfold() will pop the fold history
        # So we need to keep a copy for the prompt
        self.fold_orientations = []
        # Track the position of the punched hole and the correct choice
        self.punch_position = None
        self.correct_choice = None
        # Positions already used, so no two choices 
        # (including the correct one) can ever land on the same spot.
        self.used_positions = set()

    def fold(self, orientation):
        """Fold the test paper in the given orientation and record it."""
        self.test_paper.fold(orientation)
        self.fold_orientations.append(orientation)

    def fold_random(self, num_folds=3):
        """Fold the test paper n times in random orientations."""
        for _ in range(num_folds):
            orientation = random.choice(["north", "south", "east", "west"])
            self.fold(orientation)

    def generate_choices(self):
        """Generate five choice papers by copying the test paper, """
        """and punching a hole at a random position in each, avoiding """
        """any position that's already been used by another choice."""
        for key in self.choices:
            candidate = copy.deepcopy(self.test_paper)
            remaining = [
                (px, py)
                for py in range(self.test_paper.current_height)
                for px in range(self.test_paper.current_width)
                if (px, py) not in self.used_positions
            ]
            if not remaining:
                print("[warning] No unused positions left; reusing a position.")
                x = random.randint(0, self.test_paper.current_width - 1)
                y = random.randint(0, self.test_paper.current_height - 1)
            else:
                x, y = random.choice(remaining)
            self.used_positions.add((x, y))
            candidate.punch(x, y)
            self.choices[key] = candidate

    def punch_random(self):
        """Punch the real test paper at a random in-bounds position that """
        """hasn't already been used by one of the decoy choices."""
        remaining = [
            (px, py)
            for py in range(self.test_paper.current_height)
            for px in range(self.test_paper.current_width)
            if (px, py) not in self.used_positions
        ]
        if not remaining:
            print("[warning] No unused positions left; reusing a position.")
            x = random.randint(0, self.test_paper.current_width - 1)
            y = random.randint(0, self.test_paper.current_height - 1)
        else:
            x, y = random.choice(remaining)
        self.used_positions.add((x, y))
        self.test_paper.punch(x, y)
        self.punch_position = (x, y)
        return x, y

    def generate_answer(self):
        """Sets a random choice to be the correct answer and returns it."""
        self.test_paper.unfold()
        correct_choice = random.choice(list(self.choices.keys()))
        self.choices[correct_choice] = self.test_paper
        # Unfold all other choice papers to their original state
        for choice_paper in self.choices.values():
            if choice_paper.layer != 1:
                choice_paper.unfold()
        # Store the correct choice for later evaluation
        self.correct_choice = correct_choice
        return correct_choice

    def build_prompt(self):
        """Build the text prompt sent to the AI model for evaluation. """
        """Returns a string containing the test description, fold sequence, and choice papers."""
        lines = [
            f"A square paper with dimensions {self.test_paper.ORIGINAL_WIDTH}x"
            f"{self.test_paper.ORIGINAL_HEIGHT} is folded in this order: "
            f"{' -> '.join(self.fold_orientations)}.",
            f"After these folds, the papers dimensions are {self.folded_width}x"
            f"{self.folded_height}. "
            "A hole is then punched through all layers at one position on "
            "this folded paper. Here is the folded paper with the hole "
            "punched (1 = hole, 0 = no hole):",
            f"\n{self.folded_face}\n",
            "If this folded, punched paper were fully unfolded back to its "
            "original size, it would match exactly one of the five candidates "
            "below (A-E). 1 = hole, 0 = no hole.",
        ]
        for key, choice_paper in self.choices.items():
            lines.append(f"\nChoice {key}:")
            lines.append(choice_paper.face_to_string())
        lines.append(
            "\nWhich choice (A, B, C, D, or E) matches the paper above once "
            "fully unfolded? Respond with only the single letter."
        )
        return "\n".join(lines)

    def extract_choice(self, response_text):
        """Extract the predicted choice (A-E) from the AI model's response text."""
        """Returns the predicted choice as a single uppercase letter, or None if no valid choice is found."""
        text = response_text.strip()
        # First, look for explicit patterns indicating the final answer
        final_answer_patterns = [
            r"\\boxed\{([A-E])\}",
            r"final answer[^A-E]{0,20}([A-E])\b",
            r"answer is[:\s]*([A-E])\b",
        ]
        # If no explicit patterns are found, look for any standalone letters A-E in the text
        for pattern in final_answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).upper()
        # If no explicit patterns are found, look for any standalone letters A-E in the text
        # This is a fallback in case the model doesn't follow the expected format.
        matches = re.findall(r"\b([A-E])\b", text.upper())
        # Return the last match found, as it is likely to be the final answer.
        return matches[-1] if matches else None

    def run(self, num_folds=3, solver=None):
        """Run a single trial of the cognitive test, the solver will be the AIs API call."""
        # Perform the random folds, generate the choice papers, punch the test paper, and determine the correct answer.
        self.fold_random(num_folds)
        self.generate_choices()
        x, y = self.punch_random()
        # Snapshot the folded, punched paper here,
        # generate_answer() below unfolds test_paper back to full size.
        # build_prompt() needs to show THIS state, not the unfolded one.
        self.folded_width = self.test_paper.current_width
        self.folded_height = self.test_paper.current_height
        self.folded_face = self.test_paper.face_to_string()
        correct_choice = self.generate_answer()
        prompt = self.build_prompt()
        # Call the solver (AI model) with the generated prompt, if a solver is provided.
        solver_result = solver(prompt) if solver else None
        # Compile the results of the trial into a dictionary
        result = {
            "num_folds": num_folds,
            "fold_history": list(self.fold_orientations),
            "punch_position": (x, y),
            "correct_choice": correct_choice,
            "prompt": prompt,
        }
        # If the solver returned a result, extract the predicted choice and other relevant information.
        if solver_result:
            predicted = self.extract_choice(solver_result["text"])
            result.update({
                "raw_response": solver_result["text"],
                "predicted_choice": predicted,
                "is_correct": predicted == correct_choice,
                "elapsed_seconds": solver_result["elapsed_seconds"],
                "prompt_tokens": solver_result["prompt_tokens"],
                "output_tokens": solver_result["output_tokens"],
                "thinking_tokens": solver_result["thinking_tokens"],
                "total_tokens": solver_result["total_tokens"],
                "model_version": solver_result["model_version"],
            })
        else:
            result.update({"raw_response": None, "predicted_choice": None, "is_correct": None})
            # Return the result dictionary containing all relevant information about the trial.
        return result