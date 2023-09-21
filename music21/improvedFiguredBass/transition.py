import itertools
import logging
from collections import defaultdict
from functools import cache


class Transition:
    def __init__(self, possib_a, possib_b, segment_transition):
        self.possib_a = possib_a
        self.possib_b = possib_b

        self.segment_a = segment_transition.segment_a
        self.segment_b = segment_transition.segment_b

    @cache
    def get_cost(self, enable_logging=False):
        rulesList = self.segment_a.consecutivePossibilityRules(self.segment_a.fbRules)
        rules = []

        for rule in rulesList:
            args = []
            if len(rule) == 5:
                args = rule[-1]
            (should_run_method, method, is_correct, cost) = rule[:4]
            if should_run_method:
                rules.append((method, is_correct, cost, args))

        total_cost = 0
        for (method, isCorrect, cost, args) in rules:
            if method(self.possib_a, self.possib_b, *args) != isCorrect:
                if enable_logging:
                    logging.log(logging.INFO, f"Cost += {cost} due to {method.__name__}")
                total_cost += cost
        return total_cost

    def __repr__(self):
        def format_possibility(pos):
            return '(' + ' '.join(p.nameWithOctave.ljust(3) for p in pos) + ')'

        return f"({format_possibility(self.possib_a)} -> {format_possibility(self.possib_b)})"


class SegmentTransition:
    def __init__(self, segment_a, segment_b):
        self.segment_a = segment_a
        self.segment_b = segment_b

        self.possibs_from = self.segment_a.allCorrectSinglePossibilities()
        self.possibs_to = self.segment_b.allCorrectSinglePossibilities()

        self.transitions_matrix = defaultdict(lambda: defaultdict(Transition))

        for possib_from, possib_to in itertools.product(self.possibs_from, self.possibs_to):
            transition = Transition(possib_from, possib_to, self)
            self.transitions_matrix[possib_from][possib_to] = transition

    def get_all_transitions(self) -> list[Transition]:
        raise NotImplementedError()

    def get_transitions_from(self, possibility):
        return self.transitions_matrix[possibility].values()

    def get_transitions_to(self, possibility):
        return [d[possibility] for d in self.transitions_matrix.values()]
