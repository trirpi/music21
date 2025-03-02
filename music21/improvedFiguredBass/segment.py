# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Name:         segment.py
# Purpose:      figured bass note and notational realization.
# Authors:      Jose Cabal-Ugaz
#
# Copyright:    Copyright © 2011 Michael Scott Asato Cuthbert
# License:      BSD, see license.txt
# ------------------------------------------------------------------------------
from __future__ import annotations

import collections
import itertools

from music21 import chord
from music21 import environment
from music21 import exceptions21
from music21 import note
from music21 import pitch
from music21 import scale
from music21.improvedFiguredBass import realizer_scale
from music21.improvedFiguredBass import resolution
from music21.improvedFiguredBass.possibility import Possibility
from music21.improvedFiguredBass.rules import RuleSet
from music21.pitch import Pitch


class Segment:
    """
    A Segment corresponds to a 1:1 realization of a bassNote and notationString
    of a :class:`~music21.figuredBass.realizer.FiguredBassLine`.
    It is created by passing six arguments: a
    :class:`~music21.figuredBass.realizerScale.FiguredBassScale`, a bassNote, a notationString,
    a :class:`~music21.figuredBass.rules.Rules` object, a number of parts and a maximum pitch.
    Realizations of a Segment are represented
    as possibility tuples (see :mod:`~music21.figuredBass.possibility` for more details).

    Methods in Python's `itertools`
    module are used extensively. Methods
    which generate possibilities or possibility progressions return iterators,
    which are turned into lists in the examples
    for display purposes only.

    if fbScale is None, a realizerScale.FiguredBassScale() is created

    if fbRules is None, a rules.Rules() instance is created.  Each Segment gets
    its own deepcopy of the one given.


    Here, a Segment is created using the default values: a FiguredBassScale in C,
    a bassNote of C3, an empty notationString, and a default
    Rules object.

    >>> from music21.figuredBass import segment
    >>> s1 = segment.Segment()
    >>> s1.bassNote
    <music21.note.Note C>
    >>> s1.numParts
    4
    >>> s1.pitch_names_in_chord
    ['C', 'E', 'G']
    >>> s1.segmentChord
    <music21.chord.Chord C3 E3 G3 C4 E4 G4 C5 E5 G5>
    """
    def __init__(self,
                 bassNote: str | note.Note = 'C3',
                 notationString: str | None = None,
                 fbScale: realizer_scale.FiguredBassScale = realizer_scale.FiguredBassScale(),
                 maxPitch: str | pitch.Pitch = 'B5',
                 listOfPitches=None,
                 play_offsets=None,
                 dynamic='mf'):
        self.bassNote = note.Note(bassNote) if isinstance(bassNote, str) else bassNote
        self._maxPitch = pitch.Pitch(maxPitch) if isinstance(maxPitch, str) else maxPitch
        self.fbScale = fbScale
        self.scale_pitch_classes = set(int(p.ps) % 12 for p in self.fbScale.realizerScale.getPitches("C3", "C4"))

        self._specialResolutionRuleChecking = None
        self._singlePossibilityRuleChecking = None
        self._consecutivePossibilityRuleChecking = None

        self.prev_segment = None
        self.next_segment = None
        self.dynamic = dynamic
        self.on_beat = None
        self.melody_pitches = set()
        self.melody_pitches_at_strike = set()
        self.start_offset = 0
        self.ends_cadence = False

        self.play_offsets = play_offsets
        self.actual_notation_string = notationString
        if notationString is None:
            self.notation_strings = ['3', '6']
        elif not (last := notationString.split(",")[-1])[-1].isdigit():
            self.notation_strings = [notationString + '3']
        else:
            self.notation_strings = [notationString]

        self.segment_options = [SegmentOption(self, n, i) for i, n in enumerate(self.notation_strings)]

        self._environRules = environment.Environment('figuredBass.segment')

    @property
    def measure_number(self):
        return self.bassNote.measureNumber

    @property
    def duration(self):
        return self.bassNote.duration

    @property
    def quarterLength(self):
        return self.duration.quarterLength

    def set_pitch_names_in_chord(self):
        for option in self.segment_options:
            option.set_pitch_names_in_chord()

    def update_pitch_names_in_chord(self, past_measure):
        for option in self.segment_options:
            option.update_pitch_names_in_chord(past_measure)

    def no_melody_pitch_within_semitone(self, pitch: int):
        return (
            Pitch(pitch + 1) not in self.melody_pitches and
            Pitch(pitch - 1) not in self.melody_pitches
        )

    def get_intermediate_int_pitches(self, possib: 'Possibility', next_possib: 'Possibility'):
        """Given a possibility returns list pairs of intermediate note pitch and voice."""
        result = []
        for voice, int_pitch in enumerate(possib.integer_pitches[:-1]):
            pitch_above = int_pitch+1 if (int_pitch+1) % 12 in self.scale_pitch_classes else int_pitch+2
            result.append((pitch_above, voice))
            pitch_above = int_pitch+3 if (int_pitch+3) % 12 in self.scale_pitch_classes else int_pitch+4
            result.append((pitch_above, voice))

            pitch_below = int_pitch-1 if (int_pitch-1) % 12 in self.scale_pitch_classes else int_pitch-2
            result.append((pitch_below, voice))
            pitch_below = int_pitch-3 if (int_pitch-3) % 12 in self.scale_pitch_classes else int_pitch-4
            result.append((pitch_below, voice))

        def check(tup):
            p = tup[0]
            return (self.no_melody_pitch_within_semitone(pitch_below) and
                    p not in possib.integer_pitches and
                    p-1 not in possib.integer_pitches and
                    p+1 not in possib.integer_pitches and
                    p not in next_possib.integer_pitches and
                    p-1 not in next_possib.integer_pitches and
                    p+1 not in next_possib.integer_pitches)

        return filter(check, result)

    def resolveDominantSeventhSegment(self, segmentB):
        # noinspection PyShadowingNames
        '''
        Can resolve a Segment whose :attr:`~music21.figuredBass.segment.Segment.segmentChord`
        spells out a dominant seventh chord. If no applicable method in
        :mod:`~music21.figuredBass.resolution` can be used, the Segment is resolved
        as an ordinary Segment.

        >>> from music21.figuredBass import segment
        >>> segmentA = segment.Segment(bassNote=note.Note('G2'), notationString='7')
        >>> allDomPossib = segmentA.allCorrectSinglePossibilities()
        >>> allDomPossibList = list(allDomPossib)
        >>> len(allDomPossibList)
        8
        >>> allDomPossibList[2]
        (<music21.pitch.Pitch D4>, <music21.pitch.Pitch B3>,
         <music21.pitch.Pitch F3>, <music21.pitch.Pitch G2>)
        >>> allDomPossibList[5]
        (<music21.pitch.Pitch D5>, <music21.pitch.Pitch B4>,
         <music21.pitch.Pitch F4>, <music21.pitch.Pitch G2>)

        Here, the Soprano pitch of resolution (C6) exceeds default maxPitch of B5, so
        it's filtered out.

        >>> [p.nameWithOctave for p in allDomPossibList[7]]
        ['B5', 'F5', 'D5', 'G2']

        >>> segmentB = segment.Segment(bassNote=note.Note('C3'), notationString='')
        >>> domResPairs = segmentA.resolveDominantSeventhSegment(segmentB)
        >>> domResPairsList = list(domResPairs)
        >>> len(domResPairsList)
        7
        >>> domResPairsList[2]
        ((<music21.pitch.Pitch D4>, <...B3>, <...F3>, <...G2>),
         (<...C4>, <...C4>, <...E3>, <...C3>))
        >>> domResPairsList[5]
        ((<...D5>, <...B4>, <...F4>, <...G2>), (<...C5>, <...C5>, <...E4>, <...C3>))
        '''
        domChord = self.segmentChord
        if not domChord.isDominantSeventh():
            # Put here for stand-alone purposes.
            raise SegmentException('Dominant seventh resolution: Not a dominant seventh Segment.')
        domChordInfo = _unpackSeventhChord(domChord)
        dominantScale = scale.MajorScale().derive(domChord)
        minorScale = dominantScale.getParallelMinor()

        tonic = dominantScale.getTonic()
        subdominant = dominantScale.pitchFromDegree(4)
        majSubmediant = dominantScale.pitchFromDegree(6)
        minSubmediant = minorScale.pitchFromDegree(6)

        resChord = segmentB.segmentChord
        domInversion = (domChord.inversion() == 2)
        resInversion = (resChord.inversion())
        resolveV43toI6 = domInversion and resInversion == 1

        # if (domChord.inversion() == 0
        #     and resChord.root().name == tonic.name
        #     and (resChord.isMajorTriad() or resChord.isMinorTriad())):
        #     # "V7 to I" resolutions are always incomplete, with a missing fifth.
        #     segmentB.fbRules.forbidIncompletePossibilities = False

        dominantResolutionMethods = [
            (resChord.root().name == tonic.name and resChord.isMajorTriad(),
             resolution.dominantSeventhToMajorTonic,
             [resolveV43toI6, domChordInfo]),
            (resChord.root().name == tonic.name and resChord.isMinorTriad(),
             resolution.dominantSeventhToMinorTonic,
             [resolveV43toI6, domChordInfo]),
            ((resChord.root().name == majSubmediant.name
              and resChord.isMinorTriad()
              and domInversion == 0),
             resolution.dominantSeventhToMinorSubmediant,
             [domChordInfo]),
            ((resChord.root().name == minSubmediant.name
              and resChord.isMajorTriad()
              and domInversion == 0),
             resolution.dominantSeventhToMajorSubmediant,
             [domChordInfo]),
            ((resChord.root().name == subdominant.name
              and resChord.isMajorTriad()
              and domInversion == 0),
             resolution.dominantSeventhToMajorSubdominant,
             [domChordInfo]),
            ((resChord.root().name == subdominant.name
              and resChord.isMinorTriad()
              and domInversion == 0),
             resolution.dominantSeventhToMinorSubdominant,
             [domChordInfo])
        ]

        try:
            return self._resolveSpecialSegment(segmentB, dominantResolutionMethods)
        except SegmentException:
            self._environRules.warn(
                'Dominant seventh resolution: No proper resolution available. '
                + 'Executing ordinary resolution.')
            return self._resolveOrdinarySegment(segmentB)

    def resolveDiminishedSeventhSegment(self, segmentB, doubledRoot=False):
        # noinspection PyShadowingNames
        '''
        Can resolve a Segment whose :attr:`~music21.figuredBass.segment.Segment.segmentChord`
        spells out a diminished seventh chord. If no applicable method in
        :mod:`~music21.figuredBass.resolution` can be used, the Segment is resolved
        as an ordinary Segment.

        >>> from music21.figuredBass import segment
        >>> segmentA = segment.Segment(bassNote=note.Note('B2'), notationString='b7')
        >>> allDimPossib = segmentA.allCorrectSinglePossibilities()
        >>> allDimPossibList = list(allDimPossib)
        >>> len(allDimPossibList)
        7
        >>> [p.nameWithOctave for p in allDimPossibList[4]]
        ['D5', 'A-4', 'F4', 'B2']
        >>> [p.nameWithOctave for p in allDimPossibList[6]]
        ['A-5', 'F5', 'D5', 'B2']

        >>> segmentB = segment.Segment(bassNote=note.Note('C3'), notationString='')
        >>> dimResPairs = segmentA.resolveDiminishedSeventhSegment(segmentB)
        >>> dimResPairsList = list(dimResPairs)
        >>> len(dimResPairsList)
        7
        >>> dimResPairsList[4]
        ((<...D5>, <...A-4>, <...F4>, <...B2>), (<...E5>, <...G4>, <...E4>, <...C3>))
        >>> dimResPairsList[6]
        ((<...A-5>, <...F5>, <...D5>, <...B2>), (<...G5>, <...E5>, <...E5>, <...C3>))
        '''
        dimChord = self.segmentChord
        if not dimChord.isDiminishedSeventh():
            # Put here for stand-alone purposes.
            raise SegmentException(
                'Diminished seventh resolution: Not a diminished seventh Segment.')
        dimChordInfo = _unpackSeventhChord(dimChord)
        dimScale = scale.HarmonicMinorScale().deriveByDegree(7, dimChord.root())
        # minorScale = dimScale.getParallelMinor()

        tonic = dimScale.getTonic()
        subdominant = dimScale.pitchFromDegree(4)

        resChord = segmentB.segmentChord
        if dimChord.inversion() == 1:  # Doubled root in context
            if resChord.inversion() == 0:
                doubledRoot = True
            elif resChord.inversion() == 1:
                doubledRoot = False

        diminishedResolutionMethods = [
            (resChord.root().name == tonic.name and resChord.isMajorTriad(),
             resolution.diminishedSeventhToMajorTonic,
             [doubledRoot, dimChordInfo]),
            (resChord.root().name == tonic.name and resChord.isMinorTriad(),
             resolution.diminishedSeventhToMinorTonic,
             [doubledRoot, dimChordInfo]),
            (resChord.root().name == subdominant.name and resChord.isMajorTriad(),
             resolution.diminishedSeventhToMajorSubdominant,
             [dimChordInfo]),
            (resChord.root().name == subdominant.name and resChord.isMinorTriad(),
             resolution.diminishedSeventhToMinorSubdominant,
             [dimChordInfo])
        ]

        try:
            return self._resolveSpecialSegment(segmentB, diminishedResolutionMethods)
        except SegmentException:
            self._environRules.warn(
                'Diminished seventh resolution: No proper resolution available. '
                + 'Executing ordinary resolution.')
            return self._resolveOrdinarySegment(segmentB)

    def resolveAugmentedSixthSegment(self, segmentB):
        # noinspection PyShadowingNames
        '''
        Can resolve a Segment whose :attr:`~music21.figuredBass.segment.Segment.segmentChord`
        spells out a
        French, German, or Swiss augmented sixth chord. Italian augmented sixth Segments
        are solved as an
        ordinary Segment using :meth:`~music21.figuredBass.possibility.couldBeItalianA6Resolution`.
        If no
        applicable method in :mod:`~music21.figuredBass.resolution` can be used, the Segment
        is resolved
        as an ordinary Segment.


        >>> from music21.figuredBass import segment
        >>> segmentA = segment.Segment(bassNote=note.Note('A-2'), notationString='#6,b5,3')
        >>> segmentA.pitch_names_in_chord  # spell out a Gr+6 chord
        ['A-', 'C', 'E-', 'F#']
        >>> allAugSixthPossib = segmentA.allCorrectSinglePossibilities()
        >>> allAugSixthPossibList = list(allAugSixthPossib)
        >>> len(allAugSixthPossibList)
        7

        >>> allAugSixthPossibList[1]
        (<music21.pitch.Pitch C4>, <music21.pitch.Pitch F#3>, <...E-3>, <...A-2>)
        >>> allAugSixthPossibList[4]
        (<music21.pitch.Pitch C5>, <music21.pitch.Pitch F#4>, <...E-4>, <...A-2>)

        >>> segmentB = segment.Segment(bassNote=note.Note('G2'), notationString='')
        >>> allAugResPossibPairs = segmentA.resolveAugmentedSixthSegment(segmentB)
        >>> allAugResPossibPairsList = list(allAugResPossibPairs)
        >>> len(allAugResPossibPairsList)
        7
        >>> allAugResPossibPairsList[1]
        ((<...C4>, <...F#3>, <...E-3>, <...A-2>), (<...B3>, <...G3>, <...D3>, <...G2>))
        >>> allAugResPossibPairsList[4]
        ((<...C5>, <...F#4>, <...E-4>, <...A-2>), (<...B4>, <...G4>, <...D4>, <...G2>))
        '''
        augSixthChord = self.segmentChord
        if not augSixthChord.isAugmentedSixth():
            # Put here for stand-alone purposes.
            raise SegmentException('Augmented sixth resolution: Not an augmented sixth Segment.')
        if augSixthChord.isItalianAugmentedSixth():
            return self._resolveOrdinarySegment(segmentB)
        elif augSixthChord.isFrenchAugmentedSixth():
            augSixthType = 1
        elif augSixthChord.isGermanAugmentedSixth():
            augSixthType = 2
        elif augSixthChord.isSwissAugmentedSixth():
            augSixthType = 3
        else:
            self._environRules.warn(
                'Augmented sixth resolution: '
                + 'Augmented sixth type not supported. Executing ordinary resolution.')
            return self._resolveOrdinarySegment(segmentB)

        tonic = resolution._transpose(augSixthChord.bass(), 'M3')
        majorScale = scale.MajorScale(tonic)
        # minorScale = scale.MinorScale(tonic)
        resChord = segmentB.segmentChord
        augSixthChordInfo = _unpackSeventhChord(augSixthChord)

        augmentedSixthResolutionMethods = [
            ((resChord.inversion() == 2
              and resChord.root().name == tonic.name
              and resChord.isMajorTriad()),
             resolution.augmentedSixthToMajorTonic, [augSixthType, augSixthChordInfo]),
            ((resChord.inversion() == 2
              and resChord.root().name == tonic.name
              and resChord.isMinorTriad()),
             resolution.augmentedSixthToMinorTonic,
             [augSixthType, augSixthChordInfo]),
            ((majorScale.pitchFromDegree(5).name == resChord.bass().name
              and resChord.isMajorTriad()),
             resolution.augmentedSixthToDominant,
             [augSixthType, augSixthChordInfo])
        ]

        try:
            return self._resolveSpecialSegment(segmentB, augmentedSixthResolutionMethods)
        except SegmentException:
            self._environRules.warn(
                'Augmented sixth resolution: No proper resolution available. '
                + 'Executing ordinary resolution.')
            return self._resolveOrdinarySegment(segmentB)

    def allSinglePossibilities(self, rule_set: RuleSet):
        result = []
        for option in self.segment_options:
            result.extend(option.all_single_possibilities(rule_set))
        return result

    def all_filtered_possibilities(self, rule_set: RuleSet):
        possibs = self.allSinglePossibilities(rule_set)
        pairs = []
        for possib in possibs:
            cost = self.get_cost(rule_set, possib)
            if cost <= rule_set.MAX_SINGLE_POSSIB_COST:
                pairs.append(possib)
        return pairs

    def finish_initialization(self):
        for option in self.segment_options:
            option.finish_initialization()

    def get_cost(self, rul_set, possib):
        return rul_set.get_cost(possib, self)


class SegmentOption:
    def __init__(self, segment: Segment, notation_string: str, index: int):
        self.segment = segment
        self.notation_string = notation_string

        self.pitch_names_in_chord = None
        self.pitch_names_in_figures = None  # pitch names that are specifically in figures

        self.all_pitches_above_bass = None
        self.segment_chord = None

        self.needed_pitch_classes = set()

        self.index = index

    @property
    def bass_note(self):
        return self.segment.bassNote

    @property
    def root_note_pitch_class(self):
        return self.segment_chord.root().ps % 12

    def set_pitch_names_in_chord(self):
        self.pitch_names_in_chord = self.segment.fbScale.getPitchNames(self.segment.bassNote.pitch, self.notation_string)
        self.pitch_names_in_figures = self.segment.fbScale.getFigurePitchNames(self.segment.bassNote.pitch, self.notation_string)

    def update_pitch_names_in_chord(self, past_measure):
        self.pitch_names_in_figures = set(self.update_pitch_names_in_single_chord(self.pitch_names_in_figures, past_measure))
        self.pitch_names_in_chord = self.update_pitch_names_in_single_chord(self.pitch_names_in_chord, past_measure)

    def update_pitch_names_in_single_chord(self, pitch_names, past_measure):
        newPitchNamesInChord = []
        for name in pitch_names:
            if name in past_measure:
                newName = past_measure[name][0].modifyPitchName(name)
                newPitchNamesInChord.append(newName)
            else:
                newPitchNamesInChord.append(name)
        return newPitchNamesInChord

    def finish_initialization(self):
        self.all_pitches_above_bass = getPitches(
            self.pitch_names_in_chord, self.segment.bassNote.pitch, self.segment._maxPitch
        )
        self.segment_chord = chord.Chord(self.all_pitches_above_bass, quarterLength=self.segment.bassNote.quarterLength)

    def all_single_possibilities(self, rule_set: RuleSet):
        '''
        Returns an iterator through a set of naive possibilities for
        a Segment, using :attr:`~music21.figuredBass.segment.Segment.numParts`,
        the pitch of :attr:`~music21.figuredBass.segment.Segment.bassNote`, and
        :attr:`~music21.figuredBass.segment.Segment.allPitchesAboveBass`.

        >>> from music21.figuredBass import segment
        >>> segmentA = segment.Segment()
        >>> allPossib = segmentA.allSinglePossibilities()
        >>> allPossib.__class__
        <... 'itertools.product'>


        The number of naive possibilities is always the length of
        :attr:`~music21.figuredBass.segment.Segment.allPitchesAboveBass`
        raised to the (:attr:`~music21.figuredBass.segment.Segment.numParts` - 1)
        power. The power is 1 less than the number of parts because
        the bass pitch is constant.


        >>> allPossibList = list(allPossib)
        >>> len(segmentA.allPitchesAboveBass)
        9
        >>> segmentA.numParts
        4
        >>> len(segmentA.allPitchesAboveBass) ** (segmentA.numParts-1)
        729
        >>> len(allPossibList)
        729

        >>> for i in (81, 275, 426):
        ...    [str(p) for p in allPossibList[i]]
        ['E3', 'C3', 'C3', 'C3']
        ['C4', 'C4', 'G4', 'C3']
        ['G4', 'G3', 'C4', 'C3']
        '''
        result = []
        r = rule_set.DYNAMIC_RANGES[self.segment.dynamic]
        for i in range(r[0], r[1] + 1):
            iterables = [self.all_pitches_above_bass] * (i - 1)
            iterables.append([pitch.Pitch(self.segment.bassNote.pitch.nameWithOctave)])
            result += list(itertools.product(*iterables))
        return [Possibility(pitches, self.index) for pitches in result]

    def all_filtered_possibilities(self, rule_set: RuleSet):
        possibs = self.allSinglePossibilities(rule_set)
        pairs = []
        for possib in possibs:
            cost = self.get_cost(rule_set, possib)
            if cost <= rule_set.MAX_SINGLE_POSSIB_COST:
                pairs.append(possib)
        return pairs

    def get_cost(self, rul_set, possib):
        return rul_set.get_cost(possib, self)


def getPitches(pitchNames=('C', 'E', 'G'), bassPitch: str | pitch.Pitch = 'C3', maxPitch: str | pitch.Pitch = 'C8'):
    """
    Given a list of pitchNames, a bassPitch, and a maxPitch, returns a sorted list of
    pitches between the two limits (inclusive) which correspond to items in pitchNames.

    >>> from music21.figuredBass import segment
    >>> pitches = segment.getPitches()
    >>> print(', '.join([p.nameWithOctave for p in pitches]))
    C3, E3, G3, C4, E4, G4, C5, E5, G5, C6, E6, G6, C7, E7, G7, C8

    >>> pitches = segment.getPitches(['G', 'B', 'D', 'F'], bassPitch=pitch.Pitch('B2'))
    >>> print(', '.join([p.nameWithOctave for p in pitches]))
    B2, D3, F3, G3, B3, D4, F4, G4, B4, D5, F5, G5, B5, D6, F6, G6, B6, D7, F7, G7, B7

    >>> pitches = segment.getPitches(['F##', 'A#', 'C#'], bassPitch=pitch.Pitch('A#3'))
    >>> print(', '.join([p.nameWithOctave for p in pitches]))
    A#3, C#4, F##4, A#4, C#5, F##5, A#5, C#6, F##6, A#6, C#7, F##7, A#7

    The maxPitch must have an octave:

    >>> segment.getPitches(maxPitch=pitch.Pitch('E'))
    Traceback (most recent call last):
    ValueError: maxPitch must be given an octave
    """
    if isinstance(bassPitch, str):
        bassPitch = pitch.Pitch(bassPitch)
    if isinstance(maxPitch, str):
        maxPitch = pitch.Pitch(maxPitch)

    if maxPitch.octave is None:
        raise ValueError('maxPitch must be given an octave')
    iter1 = itertools.product(pitchNames, range(maxPitch.octave + 1))
    iter2 = map(lambda x: pitch.Pitch(x[0] + str(x[1])), iter1)
    iter3 = itertools.filterfalse(lambda samplePitch: bassPitch > samplePitch, iter2)
    iter4 = itertools.filterfalse(lambda samplePitch: samplePitch > maxPitch, iter3)
    allPitches = list(iter4)
    allPitches.sort()
    return allPitches


def _unpackSeventhChord(seventhChord):
    bass = seventhChord.bass()
    root = seventhChord.root()
    third = seventhChord.getChordStep(3)
    fifth = seventhChord.getChordStep(5)
    seventh = seventhChord.getChordStep(7)
    seventhChordInfo = [bass, root, third, fifth, seventh]
    return seventhChordInfo


def _unpackTriad(threePartChord):
    bass = threePartChord.bass()
    root = threePartChord.root()
    third = threePartChord.getChordStep(3)
    fifth = threePartChord.getChordStep(5)
    threePartChordInfo = [bass, root, third, fifth]
    return threePartChordInfo


def _compileRules(rulesList, maxLength=5):
    ruleChecking = collections.defaultdict(list)
    for ruleIndex in range(len(rulesList)):
        args = []
        if len(rulesList[ruleIndex]) == maxLength:
            args = rulesList[ruleIndex][-1]
        if maxLength == 5:
            (shouldRunMethod, method, isCorrect, cost) = rulesList[ruleIndex][0:4]
            ruleChecking[shouldRunMethod].append((method, isCorrect, cost, args))
        elif maxLength == 3:
            (shouldRunMethod, method) = rulesList[ruleIndex][0:2]
            ruleChecking[shouldRunMethod].append((method, args))

    return ruleChecking


class SegmentException(exceptions21.Music21Exception):
    pass
