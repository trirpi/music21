import itertools
from collections import defaultdict
from functools import cache

from music21.improvedFiguredBass.rules import RuleSet


class Transition:
    def __init__(self, possib_a, possib_b, segment_transition):
        self.possib_a = possib_a
        self.possib_b = possib_b

        self.segment_transition = segment_transition

    @property
    def segment_a(self):
        return self.segment_transition.segment_a

    @property
    def segment_b(self):
        return self.segment_transition.segment_b

    @cache
    def get_cost(self, enable_logging=False):
        rule_set: RuleSet = self.segment_transition.rule_set

        return rule_set.get_cost(self.possib_a, self.possib_b, {'segment_b': self.segment_b})

    def __repr__(self):
        def format_possibility(pos):
            return '(' + ' '.join(p.nameWithOctave.ljust(3) for p in pos) + ')'

        return f"({format_possibility(self.possib_a)} -> {format_possibility(self.possib_b)})"


class SegmentTransition:
    def __init__(self, segment_a, segment_b, rule_set: RuleSet):
        self.segment_a = segment_a
        self.segment_b = segment_b

        self.rule_set = rule_set

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
