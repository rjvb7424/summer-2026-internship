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
        # generate_face was made into a function so that it can be called after each fold to update
        return [[0 for _ in range(self.current_width)] for _ in range(self.current_height)]

    def fold(self, orientation = "north"):
        # Fold the paper in the specified orientation
        # Update the current dimensions, layer count,fold history and face accordingly
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
        # Print the current state of the paper's face
        # While ensuring that each row is printed on a new line
        for row in self.face:
            print(" ".join(str(cell) for cell in row))

    def punch(self, x, y):
        # Punch a hole in the paper at the specified coordinates (x, y)
        # If the coordinates are out of bounds, print an error message
        if 0 <= x < self.current_width and 0 <= y < self.current_height:
            self.face[y][x] = 1
        else:
            print("Coordinates out of bounds.")

    def unfold(self):
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
