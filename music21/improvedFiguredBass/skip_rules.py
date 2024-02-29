from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from music21.improvedFiguredBass.segment import Segment


class SkipDecision(Enum):
    UNKNOWN = 0
    SKIP = 1
    NO_SKIP = 2


class SkipRules:
    MUST_SKIP_THRESHOLD = 15

    def __init__(self):
        self.rules = [
            IsConnected(5),
            IsFast(5),
            IsAccented(5),
        ]

    def should_skip(self, segment: 'Segment') -> SkipDecision:
        if (
            segment.actual_notation_string
            or segment.duration.quarterLength >= 1
            or segment.prev_segment is None
            or segment.prev_segment.duration.quarterLength < segment.duration.quarterLength
            or segment.on_beat >= 2
        ):
            return SkipDecision.NO_SKIP

        if sum(r.get_cost(segment) for r in self.rules) >= self.MUST_SKIP_THRESHOLD:
            return SkipDecision.SKIP

        return SkipDecision.UNKNOWN


class IntermediateNoteRule(ABC):
    def __init__(self, cost):
        self.cost = cost

    @abstractmethod
    def get_cost(self, segment: 'Segment') -> int:
        pass


class IsFast(IntermediateNoteRule):
    def get_cost(self, segment: 'Segment') -> int:
        note_length = segment.bassNote.quarterLength
        if note_length == 0.5:
            return self.cost
        elif note_length < 0.5:
            return 2*self.cost
        return 0


class IsConnected(IntermediateNoteRule):
    def get_cost(self, segment: 'Segment') -> int:
        previous, next_ = segment.prev_segment.bassNote, segment.next_segment.bassNote
        note = segment.bassNote
        total = 0
        if previous is not None and self.is_connected(previous.pitch, note.pitch):
            total += self.cost
        if next_ is not None and self.is_connected(next_.pitch, note.pitch):
            total += self.cost
        return total

    @staticmethod
    def is_connected(p1, p2):
        return abs(p1.ps - p2.ps) <= 2


class IsAccented(IntermediateNoteRule):
    def get_cost(self, segment: 'Segment') -> int:
        return self.cost * (not segment.on_beat)
