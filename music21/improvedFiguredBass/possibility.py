from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from music21.pitch import Pitch
    from music21.improvedFiguredBass.segment import Segment


class Possibility:
    def __init__(self, pitches: tuple['Pitch', ...], option_index: int = 0):
        self.pitches = pitches
        self.option_index = option_index

    def __repr__(self):
        return '(' + ' '.join(p.nameWithOctave.ljust(3) for p in reversed(self.pitches)) + ')'

    def get_segment_option(self, segment: 'Segment'):
        return segment.segment_options[self.option_index]
