from itertools import permutations


class FCFSPolicy:
    def __init__(self, num_classes: int):
        ordering = tuple(range(num_classes))
        self.action = list(permutations(range(num_classes))).index(ordering)

    def select_action(self, state, valid_actions=None) -> int:
        return self.action


class FixedPriorityPolicy:
    def __init__(self, num_classes: int, acuity_weights: list[float]):
        ordering = tuple(sorted(range(num_classes), key=lambda i: acuity_weights[i], reverse=True))
        self.action = list(permutations(range(num_classes))).index(ordering)

    def select_action(self, state, valid_actions=None) -> int:
        return self.action