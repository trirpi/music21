from music21.pitch import Pitch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from music21.improvedFiguredBass.segment import Segment


class Possibility:
    def __init__(self, pitches: tuple[int | Pitch, ...], option_index: int = 0):
        if type(pitches[0]) is int:
            self.integer_pitches = pitches
        else:
            self.integer_pitches = tuple(int(pitch.ps) for pitch in pitches)
        self.option_index = option_index

    def get_pitches(self) -> tuple['Pitch', ...]:
        return tuple(Pitch(p, accidental=None) for p in self.integer_pitches)

    def __repr__(self):
        return '(' + ' '.join(p.nameWithOctave.ljust(3) for p in reversed(self.get_pitches())) + ')'

    def __eq__(self, other) -> bool:
        return self.integer_pitches == other.integer_pitches

    def __hash__(self) -> int:
        return self.integer_pitches.__hash__()

    def get_segment_option(self, segment: 'Segment'):
        return segment.segment_options[self.option_index]
