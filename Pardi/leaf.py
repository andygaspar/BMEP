import copy


class Leaf:

    def __init__(self, label, col, insertion, T_idx):
        self.label = label
        self.col = col
        self.T_idx = T_idx
        self.insertion = insertion
        self.idx = None

        self.matched = False
        self.match = None
        self.internal = [None, None]
        self.internal_list = []

        self.node = None

    def get_assignment(self, print_=False):
        assignment = self.match if self.match is not None else self.internal
        if print_:
            print(self.label, "->", assignment)
        return assignment

    def assign(self, match):
        self.match = match
        if not match.matched:
            match.node = self.insertion
            match.matched = True

    def __repr__(self):
        return self.label

    def __lt__(self, other):
        return self.col < other.col

    def __eq__(self, other):
        return self.col == other.col
