# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Name:         possibility.py
# Purpose:      rule checking functions for a "possibility" represented as a tuple.
# Authors:      Jose Cabal-Ugaz
#
# Copyright:    Copyright © 2011 Michael Scott Asato Cuthbert
# License:      BSD, see license.txt
# ------------------------------------------------------------------------------
'''
A possibility is a tuple with pitches, and is intended to encapsulate a possible
solution to a :class:`~music21.figuredBass.segment.Segment`.
Unlike a :class:`~music21.chord.Chord`,
the ordering of a possibility does matter. The assumption throughout fbRealizer
is that a possibility is always in order from the highest part to the lowest part, and
the last element of each possibility is the bass.


.. note:: fbRealizer supports voice crossing, so the order of pitches from lowest
    to highest may not correspond to the ordering of parts.


Here, a possibility is created. G5 is in the highest part, and C4 is the bass. The highest
part contains the highest Pitch, and the lowest part contains the lowest Pitch. No voice
crossing is present.

>>> G5 = pitch.Pitch('G5')
>>> C5 = pitch.Pitch('C5')
>>> E4 = pitch.Pitch('E4')
>>> C4 = pitch.Pitch('C4')
>>> p1 = (G5, C5, E4, C4)


Here, another possibility is created with the same pitches, but this time,
with voice crossing present.
C5 is in the highest part, but the highest Pitch G5 is in the second highest part.


>>> p2 = (C5, G5, E4, C4)


The methods in this module are applied to possibilities, and fall into three main categories:


1) Single Possibility Methods. These methods are applied in finding correct possibilities in
:meth:`~music21.figuredBass.segment.Segment.allCorrectSinglePossibilities`.


2) Consecutive Possibility Methods. These methods are applied to (possibA, possibB) pairs
in :meth:`~music21.figuredBass.segment.Segment.allCorrectConsecutivePossibilities`,
possibA being any correct possibility in segmentA and possibB being any correct possibility
in segmentB.


3) Special Resolution Methods. These methods are applied in
:meth:`~music21.figuredBass.segment.Segment.allCorrectConsecutivePossibilities`
as applicable if the pitch names of a Segment correctly spell out an augmented sixth, dominant
seventh, or diminished seventh chord. They are located in :mod:`~music21.figuredBass.resolution`.


The application of these methods is controlled by corresponding instance variables in a
:class:`~music21.figuredBass.rules.Rules` object provided to a Segment.



.. note:: The number of parts and maxPitch are universal for a
    :class:`~music21.figuredBass.realizer.FiguredBassLine`.
'''
from __future__ import annotations

import unittest

from music21 import exceptions21
from music21 import pitch


# SINGLE POSSIBILITY RULE-CHECKING METHODS
# ----------------------------------------
def voiceCrossing(possibA):
    '''
    Returns True if there is voice crossing present between any two parts
    in possibA. The parts from the lowest part to the highest part (right to left)
    must correspond to increasingly higher pitches in order for there to
    be no voice crossing. Comparisons between pitches are done using pitch
    comparison methods, which are based on pitch space values
    (see :class:`~music21.pitch.Pitch`).

    >>> from music21.figuredBass import possibility
    >>> C4 = pitch.Pitch('C4')
    >>> E4 = pitch.Pitch('E4')
    >>> C5 = pitch.Pitch('C5')
    >>> G5 = pitch.Pitch('G5')
    >>> possibA1 = (C5, G5, E4)
    >>> possibility.voiceCrossing(possibA1)  # G5 > C5
    True
    >>> possibA2 = (C5, E4, C4)
    >>> possibility.voiceCrossing(possibA2)
    False
    '''
    hasVoiceCrossing = False
    for part1Index in range(len(possibA)):
        higherPitch = possibA[part1Index]
        for part2Index in range(part1Index + 1, len(possibA)):
            lowerPitch = possibA[part2Index]
            if higherPitch < lowerPitch:
                hasVoiceCrossing = True
                return hasVoiceCrossing

    return hasVoiceCrossing


def isIncomplete(possibA, pitchNamesToContain):
    '''
    Returns True if possibA is incomplete, if it doesn't contain at least
    one of every pitch name in pitchNamesToContain.
    For a Segment, pitchNamesToContain is
    :attr:`~music21.figuredBass.segment.Segment.pitchNamesInChord`.


    If possibA contains excessive pitch names, a PossibilityException is
    raised, although this is not a concern with the current implementation
    of fbRealizer.

    >>> from music21.figuredBass import possibility
    >>> C3 = pitch.Pitch('C3')
    >>> E4 = pitch.Pitch('E4')
    >>> G4 = pitch.Pitch('G4')
    >>> C5 = pitch.Pitch('C5')
    >>> Bb5 = pitch.Pitch('B-5')
    >>> possibA1 = (C5, G4, E4, C3)
    >>> pitchNamesA1 = ['C', 'E', 'G', 'B-']
    >>> possibility.isIncomplete(possibA1, pitchNamesA1)  # Missing B-
    True
    >>> pitchNamesA2 = ['C', 'E', 'G']
    >>> possibility.isIncomplete(possibA1, pitchNamesA2)
    False
    '''
    isIncompleteV = False
    pitchNamesContained = []
    for givenPitch in possibA:
        if givenPitch.name not in pitchNamesContained:
            pitchNamesContained.append(givenPitch.name)
    for pitchName in pitchNamesToContain:
        if pitchName not in pitchNamesContained:
            isIncompleteV = True
    if not isIncompleteV and (len(pitchNamesContained) > len(pitchNamesToContain)):
        isIncompleteV = False
        # raise PossibilityException(str(possibA) + '
        #        contains pitch names not found in pitchNamesToContain.')

    return isIncompleteV


def upperPartsWithinLimit(possibA, maxSemitoneSeparation=12):
    '''
    Returns True if the pitches in the upper parts of possibA
    are found within maxSemitoneSeparation of each other. The
    upper parts of possibA are all the pitches except the last.

    The default value of maxSemitoneSeparation is 12 semitones,
    enharmonically equivalent to a perfect octave. If this method
    returns True for this default value, then all the notes in
    the upper parts can be played by most adult pianists using
    just the right hand.

    >>> from music21.figuredBass import possibility
    >>> C3 = pitch.Pitch('C3')
    >>> E3 = pitch.Pitch('E3')
    >>> E4 = pitch.Pitch('E4')
    >>> G4 = pitch.Pitch('G4')
    >>> C5 = pitch.Pitch('C5')
    >>> possibA1 = (C5, G4, E4, C3)
    >>> possibility.upperPartsWithinLimit(possibA1)
    True


    Here, C5 and E3 are separated by almost two octaves.


    >>> possibA2 = (C5, G4, E3, C3)
    >>> possibility.upperPartsWithinLimit(possibA2)
    False
    '''
    areUpperPartsWithinLimit = True
    if maxSemitoneSeparation is None:
        return areUpperPartsWithinLimit

    upperParts = possibA[0:len(possibA) - 1]
    for part1Index in range(len(upperParts)):
        higherPitch = upperParts[part1Index]
        for part2Index in range(part1Index + 1, len(upperParts)):
            lowerPitch = upperParts[part2Index]
            if abs(higherPitch.ps - lowerPitch.ps) > maxSemitoneSeparation:
                areUpperPartsWithinLimit = False
                return areUpperPartsWithinLimit

    return areUpperPartsWithinLimit


def pitchesWithinLimit(possibA, maxPitch=pitch.Pitch('B5')):
    '''
    Returns True if all pitches in possibA are less than or equal to
    the maxPitch provided. Comparisons between pitches are done using pitch
    comparison methods, which are based on pitch space values
    (see :class:`~music21.pitch.Pitch`).


    Used in :class:`~music21.figuredBass.segment.Segment` to filter
    resolutions of special Segments which can have pitches exceeding
    the universal maxPitch of a :class:`~music21.figuredBass.realizer.FiguredBassLine`.


    >>> from music21.figuredBass import possibility
    >>> from music21.figuredBass import resolution
    >>> G2 = pitch.Pitch('G2')
    >>> D4 = pitch.Pitch('D4')
    >>> F5 = pitch.Pitch('F5')
    >>> B5 = pitch.Pitch('B5')
    >>> domPossib = (B5, F5, D4, G2)
    >>> possibility.pitchesWithinLimit(domPossib)
    True
    >>> resPossib = resolution.dominantSeventhToMajorTonic(domPossib)
    >>> resPossib  # Contains C6 > B5
    (<music21.pitch.Pitch C6>, <music21.pitch.Pitch E5>, <music21.pitch.Pitch C4>, <music21.pitch.Pitch C3>)
    >>> possibility.pitchesWithinLimit(resPossib)
    False
    '''
    for givenPitch in possibA:
        if givenPitch > maxPitch:
            return False

    return True


def limitPartToPitch(possibA, partPitchLimits=None):
    '''
    Takes in a dict, partPitchLimits containing (partNumber, partPitch) pairs, each
    of which limits a part in possibA to a certain :class:`~music21.pitch.Pitch`.
    Returns True if all limits are followed in possibA, False otherwise.

    >>> from music21.figuredBass import possibility
    >>> C4 = pitch.Pitch('C4')
    >>> E4 = pitch.Pitch('E4')
    >>> G4 = pitch.Pitch('G4')
    >>> C5 = pitch.Pitch('C5')
    >>> G5 = pitch.Pitch('G5')
    >>> sopranoPitch = pitch.Pitch('G5')
    >>> possibA1 = (C5, G4, E4, C4)
    >>> possibility.limitPartToPitch(possibA1, {1: sopranoPitch})
    False
    >>> possibA2 = (G5, G4, E4, C4)
    >>> possibility.limitPartToPitch(possibA2, {1: sopranoPitch})
    True
    '''
    if partPitchLimits is None:
        partPitchLimits = {}
    for (partNumber, partPitch) in partPitchLimits.items():
        if not (possibA[partNumber - 1] == partPitch):
            return False

    return True


# CONSECUTIVE POSSIBILITY RULE-CHECKING METHODS
# ---------------------------------------------
# Speedup tables
PITCH_QUARTET_TO_BOOL_TYPE = dict[
    tuple[pitch.Pitch, pitch.Pitch, pitch.Pitch, pitch.Pitch],
    bool
]
parallelFifthsTable: PITCH_QUARTET_TO_BOOL_TYPE = {}
parallelOctavesTable: PITCH_QUARTET_TO_BOOL_TYPE = {}
hiddenFifthsTable: PITCH_QUARTET_TO_BOOL_TYPE = {}
hiddenOctavesTable: PITCH_QUARTET_TO_BOOL_TYPE = {}

# apply a function to one pitch of possibA at a time
# apply a function to two pitches of possibA at a time
# apply a function to one partPair of possibA, possibB at a time
# apply a function to two partPairs of possibA, possibB at a time
# use an iterator that fails when the first false is returned


singlePossibilityMethods = [voiceCrossing, isIncomplete, upperPartsWithinLimit, pitchesWithinLimit]


# singlePossibilityMethods.sort(None, lambda x: x.__name__)


class PossibilityException(exceptions21.Music21Exception):
    pass


# ------------------------------------------------------------------------------

class Test(unittest.TestCase):
    pass


if __name__ == '__main__':
    import music21

    music21.mainTest(Test)
