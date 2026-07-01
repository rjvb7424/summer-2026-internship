import random
import copy
import re

class Paper():
    def __init__(self, current_width = 10, current_height = 10, layer = 1):
        # Original dimensions of the paper
        self.ORIGINAL_WIDTH = current_width
        self.ORIGINAL_HEIGHT = current_height
        # Current dimensions of the paper
        self.current_width = current_width
        self.current_height = current_height
        # 2D grid representing the current face of the paper
        self.face = self.generate_face()
        # History of folds made on the paper as a list of orientations
        self.fold_history = []
        # Number of layers of paper after folding
        self.layer = layer

    def generate_face(self):
        """Generate a 2D grid representing the current face of the paper based on its current dimensions."""
        return [[0 for _ in range(self.current_width)] for _ in range(self.current_height)]
    
    def face_to_string(self):
        """Convert the current face of the paper to a string representation."""
        return "\n".join(" ".join(str(c) for c in row) for row in self.face)

    def fold(self, orientation = "north"):
        """Fold the paper in the specified orientation.
        Update the current dimensions, layer count, fold history and face accordingly."""
        match orientation:
            case "north":
                self.current_height = self.current_height // 2
                self.layer = self.layer * 2
                self.fold_history.append("north")
                self.face = self.generate_face()
            case "south":
                self.current_height = self.current_height // 2
                self.layer = self.layer * 2
                self.fold_history.append("south")
                self.face = self.generate_face()
            case "east":
                self.current_width = self.current_width // 2
                self.layer = self.layer * 2
                self.fold_history.append("east")
                self.face = self.generate_face()
            case "west":
                self.current_width = self.current_width // 2
                self.layer = self.layer * 2
                self.fold_history.append("west")
                self.face = self.generate_face()

    def visualize(self):
        """Print the current state of the paper's face."""
        # While ensuring that each row is printed on a new line
        for row in self.face:
            print(" ".join(str(cell) for cell in row))

    def punch(self, x, y):
        """Punch a hole in the paper at the specified coordinates (x, y).
        If the coordinates are out of bounds, print an error message"""
        if 0 <= x < self.current_width and 0 <= y < self.current_height:
            self.face[y][x] = 1
        else:
            print("Coordinates out of bounds.")

    def unfold(self):
        """Unfold the paper back to its original state."""
        while self.layer != 1:
            last_fold = self.fold_history.pop()
            if last_fold == "north":
                # Current face was the north half; mirror it downward for the south half
                new_face = self.face + self.face[::-1]
            elif last_fold == "south":
                # Current face was the south half; mirror it upward for the north half
                new_face = self.face[::-1] + self.face
            elif last_fold == "west":
                # Current face was the west half; mirror it rightward for the east half
                new_face = [row + row[::-1] for row in self.face]
            elif last_fold == "east":
                # Current face was the east half; mirror it leftward for the west half
                new_face = [row[::-1] + row for row in self.face]
            # Update the current dimensions and layer count based on the last fold
            if last_fold in ("north", "south"):
                self.current_height *= 2
            else:
                self.current_width *= 2
            self.face = new_face
            self.layer //= 2
        return
    
class CognitiveTest():
    def __init__(self):
        # Test paper for the cognitive test
        self.test_paper = Paper()
        # Possible choices to pick from
        self.choices = {"A": None, "B": None, "C": None, "D": None, "E": None}

    def generate_choices(self, test_paper=None):
        """Populate the choices dictionary with the current state of the paper after folding"""
        for key in self.choices.keys():
            self.choices[key] = copy.deepcopy(test_paper)
        # Punch a hole at a random position on the choices paper
        for key, paper in self.choices.items():
            x = random.randint(0, self.test_paper.current_width - 1)
            y = random.randint(0, self.test_paper.current_height - 1)
            paper.face[y][x] = 1
    
    def generate_answer(self, test_paper=None):
        """Overwrite a random choice with the correct unfolded state of the paper and return the key of that choice"""
        correct_choice = random.choice(list(self.choices.keys()))
        self.choices[correct_choice] = test_paper
        return correct_choice

    def run(self, num_folds=3):
        """Run the cognitive test by folding the paper, punching a hole, 
        and asking to identify the correct unfolded state."""
        print("Initial paper (unfolded):")
        self.test_paper.visualize()
        # Fold the paper n times in a random oritentation
        for i in range(num_folds):
            orientation = random.choice(["north", "south", "east", "west"])
            self.test_paper.fold(orientation)
            print(f"Step {i+1}: Folded {orientation}. Current state of the paper:")
            self.test_paper.visualize()
        # After folding the test paper, generate the choices for the cognitive test
        self.generate_choices(test_paper=self.test_paper)
        # Punch a hole at a random position on the folded paper
        x = random.randint(0, self.test_paper.current_width - 1)
        y = random.randint(0, self.test_paper.current_height - 1)
        print(f"Punching a hole at position ({x}, {y}) through all {self.test_paper.layer} layer(s):")
        self.test_paper.punch(x, y)
        self.test_paper.visualize()
        # Generate the correct answer choice
        self.test_paper.unfold()
        correct_choice = self.generate_answer(test_paper=self.test_paper)
        print("Which of these choices (A, B, C, D, E) represents the correct unfolded state of the paper?")
        for key, paper in self.choices.items():
            print(f"Choice {key}:")
            paper.unfold()
            paper.visualize()
        while input("Enter your choice (A, B, C, D, E): ").upper() != correct_choice:
            print("Incorrect!")
        print(f"Correct!")

    def build_prompt(self):
        """Build a prompt for the cognitive test, which will be sent to the AI models for evaluation."""
        lines = [
            f"A square paper with dimensions {self.test_paper.current_width}x{self.test_paper.current_height}" 
            "is folded several times, a hole is punched through all layers, then it is unfolded.",
            f"Fold sequence: {' -> '.join(self.test_paper.fold_history_log)}",
            "\nBelow are five candidates for the possible unfolded state (A-E). 1 = hole, 0 = no hole.",
        ]
        for key, choice_paper in self.choices.items():
            lines.append(f"\nChoice {key}:")
            lines.append(choice_paper.face_to_string())
        lines.append("\nWhich choice (A, B, C, D, or E) is correct? Respond with only the single letter.")
        return "\n".join(lines)

    def extract_choice(self, response_text):
        """Extract the choice (A-E) from the response text, if present."""
        """ This is done because the model might return additional text"""
        match = re.search(r"\b([A-E])\b", response_text.strip().upper())
        return match.group(1) if match else None

    def run_automated(self, num_folds=3, solver=None):
        """Run the cognitive test in an automated manner, 
        optionally using a solver function to evaluate the prompt and return a predicted choice."""
        # Fold the paper n times in a random orientation and record the orientations
        fold_orientations = []
        for _ in range(num_folds):
            orientation = random.choice(["north", "south", "east", "west"])
            self.test_paper.fold(orientation)
            fold_orientations.append(orientation)
        # After folding the test paper, generate the choices for the cognitive test
        self.generate_choices(test_paper=self.test_paper)
        # Punch a hole at a random position on the folded paper
        x = random.randint(0, self.test_paper.current_width - 1)
        y = random.randint(0, self.test_paper.current_height - 1)
        self.test_paper.punch(x, y)
        # Unfold the test paper to determine the correct choice
        self.test_paper.unfold()
        correct_choice = self.generate_answer(test_paper=self.test_paper)
        # Unfold the distractor choices so all five are comparable
        for choice_paper in self.choices.values():
            if choice_paper.layer != 1:
                choice_paper.unfold()
        # Build the prompt for the cognitive test, which will be sent to the AI models for evaluation
        prompt = self.build_prompt()
        solver_result = solver(prompt) if solver else None
        # Compile the results of the cognitive test
        result = {
            "num_folds": num_folds,
            "fold_history": fold_orientations,
            "punch_position": (x, y),
            "correct_choice": correct_choice,
            "prompt": prompt,
        }
        # If a solver was provided and returned a result, extract the predicted choice and other relevant information
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
        # Return the compiled results of the cognitive test
        return result
