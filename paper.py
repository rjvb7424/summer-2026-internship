class Paper():
    def __init__(self, current_width = 10, current_height = 10, layer = 1):
        # Original dimensions of the paper
        self.ORIGINAL_WIDTH = current_width
        self.ORIGINAL_HEIGHT = current_height
        # Current dimensions of the paper
        self.current_width = current_width
        self.current_height = current_height
        # 2D grid representing the current face of the paper
        self.face = [[0 for _ in range(current_width)] for _ in range(current_height)]
        # History of folds made on the paper as a list of orientations
        self.fold_history = []
        # Number of layers of paper after folding
        self.layer = layer

    def fold(self, orientation = "north"):
        match orientation:
            case "north":
                self.current_height = self.current_height // 2
                self.layer = self.layer * 2
                self.fold_history.append("north")
            case "south":
                self.current_height = self.current_height // 2
                self.layer = self.layer * 2
                self.fold_history.append("south")
            case "east":
                self.current_width = self.current_width // 2
                self.layer = self.layer * 2
                self.fold_history.append("east")
            case "west":
                self.current_width = self.current_width // 2
                self.layer = self.layer * 2
                self.fold_history.append("west")

    def visualize(self):
        # Print the current state of the paper's face
        # While ensuring that each row is printed on a new line
        for row in self.face:
            print(" ".join(str(cell) for cell in row))
