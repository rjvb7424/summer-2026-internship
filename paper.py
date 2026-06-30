class Paper():
    def __init__(self, width = 10, height = 10, planes = 1):
        self.width = width
        self.height = height
        self.planes = planes
        self.fold_history = []

    def fold(self, orientation = "north"):
        match orientation:
            case "north":
                self.height = self.height / 2
                self.planes = self.planes * 2
                self.fold_history.append("north")
            case "south":
                self.height = self.height / 2
                self.planes = self.planes * 2
                self.fold_history.append("south")
            case "east":
                self.width = self.width / 2
                self.planes = self.planes * 2
                self.fold_history.append("east")
            case "west":
                self.width = self.width / 2
                self.planes = self.planes * 2
                self.fold_history.append("west")
