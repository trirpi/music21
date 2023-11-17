from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from music21.pitch import Pitch


class Possibility:
    def __init__(self, pitches: tuple['Pitch']):
        self.pitches = pitches

    def __repr__(self):
        return '(' + ' '.join(p.nameWithOctave.ljust(3) for p in reversed(self.pitches)) + ')'
