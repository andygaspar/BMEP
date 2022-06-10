

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

    def assign_internals(self):
        sorted_internal = sorted(self.internal_list, reverse=True)
        for leaf in sorted_internal:
            for i in range(len(sorted_internal)):
                if leaf == self.internal_list[i]:
                    if i == 0:
                        leaf.internal[0] = self.idx
                    else:
                        leaf.internal[0] = self.internal_list[i - 1].insertion
                    if i < len(self.internal_list) - 1:
                        leaf.internal[1] = self.internal_list[i + 1].insertion
                    else:
                        leaf.internal[1] = self.insertion

                    self.internal_list.remove(leaf)
                    break

    def assign(self, match):
        self.match = match
        if not match.matched:
            match.node = self.insertion
            match.matched = True
        if self.internal_list:
            self.assign_internals()

    def __repr__(self):
        return self.label

    def __lt__(self, other):
        return self.col < other.col

    def __eq__(self, other):
        return self.col == other.col
