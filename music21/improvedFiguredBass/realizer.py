# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Name:         realizer.py
# Purpose:      figured bass lines, consisting of notes
#                and figures in a given key.
# Authors:      Jose Cabal-Ugaz
#
# Copyright:    Copyright © 2011 Michael Scott Asato Cuthbert
# License:      BSD, see license.txt
# ------------------------------------------------------------------------------
"""
This module, the heart of fbRealizer, is all about realizing
a bass line of (bassNote, notationString)
pairs. All it takes to create well-formed realizations of a
bass line is a few lines of music21 code,
from start to finish. See :class:`~music21.figuredBass.realizer.FiguredBassLine` for more details.

>>> from music21.improvedFiguredBass import realizer
>>> fbLine = realizer.FiguredBassLine()
>>> fbLine.add_element(note.Note('C3'))
>>> fbLine.add_element(note.Note('D3'), '4,3')
>>> fbLine.add_element(note.Note('C3', quarterLength = 2.0))
>>> allSols = fbLine.realize()
"""
from __future__ import annotations

import copy
import logging
import typing as t
import unittest

from tqdm import tqdm

from music21 import chord
from music21 import clef
from music21 import exceptions21
from music21 import key
from music21 import meter
from music21 import note
from music21 import pitch
from music21 import stream
from music21.figuredBass import checker
from music21.improvedFiguredBass import notation
from music21.improvedFiguredBass import realizer_scale
from music21.improvedFiguredBass import rules_config
from music21.improvedFiguredBass import segment
from music21.improvedFiguredBass.helpers import format_possibility
from music21.improvedFiguredBass.rules import RuleSet
from music21.improvedFiguredBass.rules_config import RulesConfig
from music21.improvedFiguredBass.skip_rules import SkipDecision
from music21.improvedFiguredBass.transition import SegmentTransition


def figured_bass_from_stream(stream_part: stream.Stream) -> FiguredBassLine:
    sf = stream_part.flatten()
    sfn = sf.getElementsByClass(note.Note)
    myKey: key.Key
    if firstKey := sf[key.Key].first():
        myKey = firstKey
    elif firstKeySignature := sf[key.KeySignature].first():
        myKey = firstKeySignature.asKey('major')
    else:
        myKey = key.Key('C')

    ts: meter.TimeSignature
    if first_ts := sf[meter.TimeSignature].first():
        ts = first_ts
    else:
        ts = meter.TimeSignature('4/4')

    fb = FiguredBassLine(myKey, ts)
    if stream_part.hasMeasures():
        m_first = stream_part.measure(0, indicesNotNumbers=True)
        if t.TYPE_CHECKING:
            assert m_first is not None
        paddingLeft = m_first.paddingLeft
        if paddingLeft != 0.0:
            fb._paddingLeft = paddingLeft

    # noinspection PyShadowingNames
    def update_annotation_string(annotationString: str, inputText: str) -> str:
        """
        Continue building the working `annotationString` based on some `inputText`
        that has yet to be processed. Called recursively until `inputText` is exhausted
        or contains unexpected characters.
        """
        # "64" and "#6#42" but not necessarily "4-3" or "sus4"
        if inputText[0] in '+#bn' and len(inputText) > 1 and inputText[1].isnumeric():
            stop_index_exclusive = 2
        elif inputText[0].isnumeric():
            stop_index_exclusive = 1
        else:
            stop_index_exclusive = 1000
        annotationString += inputText[:stop_index_exclusive]
        # Is there more?
        if inputText[stop_index_exclusive:]:
            annotationString += ', '
            annotationString = update_annotation_string(
                annotationString, inputText[stop_index_exclusive:])
        return annotationString

    for n in sfn:
        if n.lyrics:
            annotationString: str = ''
            for i, lyric_line in enumerate(n.lyrics):
                if lyric_line.text in (None, ''):
                    continue
                if ',' in lyric_line.text:
                    # presence of comma suggests we already have a separated
                    # sequence of figures, e.g. "#6, 4, 2"
                    annotationString = lyric_line.text
                else:
                    # parse it more carefully
                    annotationString = update_annotation_string(annotationString, lyric_line.text)
                if i + 1 < len(n.lyrics):
                    annotationString += ', '
            if annotationString == "=":
                fb._fbList[-1][0].duration.addDurationTuple(copy.deepcopy(n.duration))
            else:
                fb.add_element(copy.deepcopy(n), annotationString)
        else:
            fb.add_element(copy.deepcopy(n))

    return fb


def add_lyrics_to_bass_note(bass_note, notation_string=None):
    """
    Takes in a bassNote and a corresponding notationString as arguments.
    Adds the parsed notationString as lyrics to the bassNote, which is
    useful when displaying the figured bass in external software.

    >>> from music21.improvedFiguredBass import realizer
    >>> n1 = note.Note('G3')
    >>> realizer.add_lyrics_to_bass_note(n1, '6,4')
    >>> n1.lyrics[0].text
    '6'
    >>> n1.lyrics[1].text
    '4'
    >>> #_DOCS_SHOW n1.show()

    .. image:: images/figuredBass/fbRealizer_lyrics.*
        :width: 100
    """
    bass_note.lyrics = []
    n = notation.Notation(notation_string)
    if not n.figureStrings:
        return
    maxLength = 0
    for fs in n.figureStrings:
        if len(fs) > maxLength:
            maxLength = len(fs)
    for fs in n.figureStrings:
        spacesInFront = ''
        for i in range(maxLength - len(fs)):
            spacesInFront += ' '
        bass_note.addLyric(spacesInFront + fs, applyRaw=True)


def _trim_all_movements(segment_list):
    """
    Each :class:`~music21.figuredBass.segment.Segment` which resolves to another
    defines a list of movements, nextMovements. Keys for nextMovements are correct
    single possibilities of the current Segment. For a given key, a value is a list
    of correct single possibilities in the subsequent Segment representing acceptable
    movements between the two. There may be movements in a string of Segments which
    directly or indirectly lead nowhere. This method is designed to be called on
    a list of Segments **after** movements are found, as happens in
    :meth:`~music21.figuredBass.realizer.FiguredBassLine.realize`.
    """
    if len(segment_list) < 3:
        return

    segment_list.reverse()
    # gets this wrong...  # pylint: disable=cell-var-from-loop
    movementsAB = None
    for segmentIndex in range(1, len(segment_list) - 1):
        movementsAB = segment_list[segmentIndex + 1].movements
        movementsBC = segment_list[segmentIndex].movements
        # eliminated = []
        for (possibB, possibCList) in list(movementsBC.items()):
            if not possibCList:
                del movementsBC[possibB]
        for (possibA, possibBList) in list(movementsAB.items()):
            movementsAB[possibA] = list(
                filter(lambda possibBB: (possibBB in movementsBC), possibBList))

    for (possibA, possibBList) in list(movementsAB.items()):
        if not possibBList:
            del movementsAB[possibA]

    segment_list.reverse()


class FiguredBassLine:
    def __init__(self, inKey=None, inTime=None):
        if inKey is None:
            inKey = key.Key('C')
        if inTime is None:
            inTime = meter.TimeSignature('4/4')

        self.inKey = inKey
        self.inTime = inTime
        self._paddingLeft = 0.0
        self._overlaidParts = stream.Part()
        self._fbScale = realizer_scale.FiguredBassScale(inKey.pitchFromDegree(1), inKey.mode)
        self._fbList = []

    def add_element(self, bassObject: note.Note, notationString=None):
        """
        Use this method to add (bassNote, notationString) pairs to the bass line. Elements
        are realized in the order they are added.


        >>> from music21.improvedFiguredBass import realizer
        >>> fbLine = realizer.FiguredBassLine(key.Key('B'), meter.TimeSignature('3/4'))
        >>> fbLine.add_element(note.Note('B2'))
        >>> fbLine.add_element(note.Note('C#3'), '6')
        >>> fbLine.add_element(note.Note('D#3'), '6')
        """
        bassObject.editorial.notationString = notationString
        c = bassObject.classes
        if 'Note' not in c:
            raise FiguredBassLineException(
                f'Not a valid bassObject (only note.Note supported) was {bassObject!r}')
        self._fbList.append((bassObject, notationString))  # a bass note, and a notationString
        add_lyrics_to_bass_note(bassObject, notationString)

    def generate_bass_line(self):
        bassLine = stream.Part()
        bassLine.append(clef.BassClef())
        bassLine.append(key.KeySignature(self.inKey.sharps))
        bassLine.append(copy.deepcopy(self.inTime))
        r = None
        if self._paddingLeft != 0.0:
            r = note.Rest(quarterLength=self._paddingLeft)
            bassLine.append(r)

        for (bassNote, unused_notationString) in self._fbList:
            bassLine.append(bassNote)

        bl2 = bassLine.makeNotation(inPlace=False, cautionaryNotImmediateRepeat=False)
        if r is not None:
            m0 = bl2.getElementsByClass(stream.Measure).first()
            m0.remove(m0.getElementsByClass(note.Rest).first())
            m0.padAsAnacrusis()
        return bl2

    def retrieve_segments(self, fbRules=None, maxPitch=None):
        '''
        generates the segmentList from an fbList, including any overlaid Segments

        if fbRules is None, creates a new rules.Rules() object

        if maxPitch is None, uses pitch.Pitch('B5')
        '''
        if fbRules is None:
            fbRules = rules_config.RulesConfig()
        if maxPitch is None:
            maxPitch = pitch.Pitch('B5')
        segmentList = []
        bassLine = self.generate_bass_line()
        if self._overlaidParts:
            self._overlaidParts.append(bassLine)
            currentMapping = checker.extractHarmonies(self._overlaidParts)
        else:
            currentMapping = checker.createOffsetMapping(bassLine)
        allKeys = sorted(currentMapping.keys())
        bassLine = bassLine.flatten().notes
        bassNoteIndex = 0
        previousBassNote = bassLine[bassNoteIndex]
        bassNote = currentMapping[allKeys[0]][-1]
        play_offsets = allKeys[0]
        previousSegment = segment.Segment(bassNote, bassNote.editorial.notationString,
                                          self._fbScale,
                                          fbRules, maxPitch, play_offsets=play_offsets)
        previousSegment.quarterLength = previousBassNote.quarterLength
        segmentList.append(previousSegment)
        for k in allKeys[1:]:
            (startTime, unused_endTime) = k
            bassNote = currentMapping[k][-1]
            currentSegment = segment.Segment(bassNote, bassNote.editorial.notationString,
                                             self._fbScale,
                                             fbRules, maxPitch, play_offsets=k)
            for partNumber in range(1, len(currentMapping[k])):
                upperPitch = currentMapping[k][partNumber - 1]
                currentSegment.rules_config._partPitchLimits.append((partNumber, upperPitch))
            if startTime == previousBassNote.offset + previousBassNote.quarterLength:
                bassNoteIndex += 1
                previousBassNote = bassLine[bassNoteIndex]
                currentSegment.quarterLength = previousBassNote.quarterLength
            else:
                # Fictitious, representative only for harmonies preserved
                # with addition of melody or melodies
                currentSegment.quarterLength = 0.0
            segmentList.append(currentSegment)
            previousSegment = currentSegment
        return segmentList

    # noinspection PyUnreachableCode
    def realize(self, fb_rules=None, max_pitch=None, rule_set=None, start_offset=0):
        # noinspection PyShadowingNames
        """
        Creates a :class:`~music21.figuredBass.segment.Segment`
        for each (bassNote, notationString) pair
        added using :meth:`~music21.figuredBass.realizer.FiguredBassLine.addElement`.
        Each Segment is associated
        with the :class:`~music21.figuredBass.rules.Rules` object provided, meaning that rules are
        universally applied across all Segments. The number of parts in a realization
        (including the bass) can be controlled through numParts, and the maximum pitch can
        likewise be controlled through maxPitch.
        Returns a :class:`~music21.figuredBass.realizer.Realization`.

        If this method is called without having provided any (bassNote, notationString) pairs,
        a FiguredBassLineException is raised. If only one pair is provided, the Realization will
        contain :meth:`~music21.figuredBass.segment.Segment.allCorrectConsecutivePossibilities`
        for the one note.

        if `fbRules` is None, creates a new rules.Rules() object

        if `maxPitch` is None, uses pitch.Pitch('B5')

        >>> from music21.improvedFiguredBass import realizer
        >>> from music21.improvedFiguredBass import rules
        >>> fbLine = realizer.FiguredBassLine(key.Key('B'), meter.TimeSignature('3/4'))
        >>> fbLine.add_element(note.Note('B2'))
        >>> fbLine.add_element(note.Note('C#3'), '6')
        >>> fbLine.add_element(note.Note('D#3'), '6')
        >>> fbRules = rules.RulesConfig()
        >>> r1 = fbLine.realize(fb_rules)
        """
        if fb_rules is None:
            fb_rules = rules_config.RulesConfig()
        if max_pitch is None:
            max_pitch = pitch.Pitch('B5')

        segmentList = self.retrieve_segments(fb_rules, max_pitch)
        if not segmentList:
            raise FiguredBassLineException('No (bassNote, notationString) pairs to realize.')

        return Realization(realizedSegmentList=segmentList, inKey=self.inKey,
                           inTime=self.inTime, overlaidParts=self._overlaidParts[0:-1],
                           paddingLeft=self._paddingLeft, rule_set=rule_set, start_offset=start_offset)


class Realization:
    """
    Returned by :class:`~music21.figuredBass.realizer.FiguredBassLine` after calling
    :meth:`~music21.figuredBass.realizer.FiguredBassLine.realize`. Allows for the
    generation of realizations as a :class:`~music21.stream.Score`.
    """
    def __init__(self, **fb_line_outputs):
        # fbLineOutputs always will have three elements, checks are for sphinx documentation only.
        if 'realizedSegmentList' in fb_line_outputs:
            self.segment_list = fb_line_outputs['realizedSegmentList']
        if 'inKey' in fb_line_outputs:
            self._inKey = fb_line_outputs['inKey']
            self._keySig = key.KeySignature(self._inKey.sharps)
        if 'inTime' in fb_line_outputs:
            self._inTime = fb_line_outputs['inTime']
        if 'overlaidParts' in fb_line_outputs:
            self._overlaidParts = fb_line_outputs['overlaidParts']
        if 'paddingLeft' in fb_line_outputs:
            self._paddingLeft = fb_line_outputs['paddingLeft']
        if 'rule_set' in fb_line_outputs:
            self.rule_set = fb_line_outputs['rule_set']
        else:
            self.rule_set = RuleSet(RulesConfig())
        if 'start_offset' in fb_line_outputs:
            self.start_offset = fb_line_outputs['start_offset']
        else:
            self.start_offset = 0
        self.keyboardStyleOutput = True

    def generate_dp_table(self):
        first_possibilities = self.segment_list[0].all_filtered_possibilities(self.rule_set)

        dp: list[dict] = [  # (possibility, cost) dicts for each segment index i
            {possib: self.segment_list[0].get_cost(self.rule_set, possib) for possib in first_possibilities}
        ]

        for i in range(len(self.segment_list) - 1):
            segment_b = self.segment_list[i + 1]
            possibs_to = segment_b.all_filtered_possibilities(self.rule_set)
            dp_entry = {}
            for possib in tqdm(possibs_to, leave=False, desc=f"Segment {i + 1}/{len(self.segment_list)}"):
                best_cost = float('inf')
                for segment_a_idx in range(i, max(-1, i-4), -1):
                    num_skips = i - segment_a_idx
                    segment_a = self.segment_list[segment_a_idx]
                    skip_decision = self.rule_set.should_skip(segment_a)
                    if skip_decision == SkipDecision.SKIP:
                        continue
                    transition = SegmentTransition(segment_a, segment_b, self.rule_set)
                    for prev_possib, prev_cost in dp[segment_a_idx].items():

                        transition_cost = transition.transitions_matrix[prev_possib][possib].get_cost()
                        local_cost_b = segment_b.get_cost(self.rule_set, possib)
                        new_cost = prev_cost + (num_skips+1)*(transition_cost + local_cost_b)

                        best_cost = min(new_cost, best_cost)
                    if skip_decision == SkipDecision.NO_SKIP:
                        break

                dp_entry[possib] = best_cost
            dp.append(dp_entry)
        return dp

    def get_reverse_choices(self, dp):
        # Extract best possib
        reverse_progression = []
        d = dp[-1]
        best_possib = min(d, key=d.get)
        final_cost = min(d.values())
        best_cost = final_cost
        reverse_progression.append((best_possib, 0))

        logging.log(logging.INFO, f"======================================")
        logging.log(logging.INFO, f"Found solution with cost {final_cost}.")
        logging.log(logging.INFO, f"======================================")
        i = len(self.segment_list) - 2
        while i >= 0:
            segment_b = self.segment_list[i + 1]
            for segment_a_idx in range(i, max(-1, i-4), -1):
                num_skips = i - segment_a_idx
                segment_a = self.segment_list[segment_a_idx]
                transition = SegmentTransition(segment_a, segment_b, self.rule_set)
                found = False
                for possib_a, prev_cost in dp[segment_a_idx].items():
                    to_possib_cost = transition.segment_b.get_cost(self.rule_set, best_possib)
                    transition_cost = transition.transitions_matrix[possib_a][best_possib].get_cost()
                    if (num_skips+1) * (transition_cost + to_possib_cost) == best_cost - prev_cost:
                        best_possib = possib_a
                        best_cost = prev_cost
                        reverse_progression.append((best_possib, num_skips))
                        logging.log(logging.INFO, f"Chose {format_possibility(best_possib)} after skipping {num_skips} notes.")
                        i -= num_skips
                        found = True
                        break

                if found:
                    break
            i -= 1
        return reverse_progression

    def get_optimal_possibility_progression(self):
        """
        Returns a random unique possibility progression.
        """
        dp = self.generate_dp_table()
        reverse_progression = self.get_reverse_choices(dp)

        # Delete skipped notes
        curr_idx = 0
        result = []
        idx_to_delete = []
        prev_segment = None
        prev_val = None
        measure = float('inf')
        logging.log(logging.INFO, f"=== START LOG ========================\n")
        for val, num_skips in reversed(reverse_progression):
            curr_segment = self.segment_list[curr_idx]
            if (m := curr_segment.measure_number) < measure:
                measure = m
                logging.log(logging.INFO, f"### Measure {measure} ###")
            if prev_val:
                self.rule_set.get_cost(val, curr_segment, prev_val, prev_segment, enable_logging=True)

            self.rule_set.get_cost(val, curr_segment, enable_logging=True)
            result.append(val)
            for i in range(curr_idx+1, curr_idx+1+num_skips):
                self.segment_list[curr_idx].quarterLength += self.segment_list[i].quarterLength
                idx_to_delete.append(i)
            curr_idx += 1 + num_skips
            prev_segment = curr_segment
            prev_val = val

        for i in reversed(idx_to_delete):
            del self.segment_list[i]


        return result

    def generate_realization_from_possibility_progression(self, possibility_progression):
        """
        Generates a realization as a :class:`~music21.stream.Score` given a possibility progression.
        """
        sol = stream.Score()

        bassLine = stream.Part()
        bassLine.append([copy.deepcopy(self._keySig), copy.deepcopy(self._inTime)])
        r = None
        if self._paddingLeft != 0.0:
            r = note.Rest(quarterLength=self._paddingLeft)
            bassLine.append(copy.deepcopy(r))

        if self.keyboardStyleOutput:
            rightHand = stream.Part()
            sol.insert(0.0, rightHand)
            rightHand.append([copy.deepcopy(self._keySig), copy.deepcopy(self._inTime)])
            if r is not None:
                rightHand.append(copy.deepcopy(r))

            for segmentIndex in range(len(self.segment_list)):
                possibA = possibility_progression[segmentIndex]
                bassNote = self.segment_list[segmentIndex].bassNote
                bassLine.append(copy.deepcopy(bassNote))
                rhPitches = possibA[0:-1]
                rhChord = chord.Chord(rhPitches)
                rhChord.quarterLength = self.segment_list[segmentIndex].quarterLength
                rightHand.append(rhChord)
            rightHand.insert(0.0, clef.TrebleClef())

            rightHand.makeNotation(inPlace=True, cautionaryNotImmediateRepeat=False)
            if r is not None:
                rightHand[0].pop(3)
                rightHand[0].padAsAnacrusis()

        else:  # Chorale-style output
            upperParts = []
            for partNumber in range(len(possibility_progression[0]) - 1):
                fbPart = stream.Part()
                sol.insert(0.0, fbPart)
                fbPart.append([copy.deepcopy(self._keySig), copy.deepcopy(self._inTime)])
                if r is not None:
                    fbPart.append(copy.deepcopy(r))
                upperParts.append(fbPart)

            for segmentIndex in range(len(self.segment_list)):
                possibA = possibility_progression[segmentIndex]
                bassNote = self.segment_list[segmentIndex].bassNote
                bassLine.append(copy.deepcopy(bassNote))

                for partNumber in range(len(possibA) - 1):
                    n1 = note.Note(possibA[partNumber])
                    n1.quarterLength = self.segment_list[segmentIndex].quarterLength
                    upperParts[partNumber].append(n1)

            for upperPart in upperParts:
                c = clef.bestClef(upperPart, allowTreble8vb=True, recurse=True)
                upperPart.insert(0.0, c)
                upperPart.makeNotation(inPlace=True, cautionaryNotImmediateRepeat=False)
                if r is not None:
                    upperPart[0].pop(3)
                    upperPart[0].padAsAnacrusis()

        bassLine.insert(0.0, clef.BassClef())
        bassLine.makeNotation(inPlace=True, cautionaryNotImmediateRepeat=False)
        if r is not None:
            bassLine[0].pop(3)
            bassLine[0].padAsAnacrusis()
        sol.insert(0.0, bassLine)
        return sol

    def generate_optimal_realization(self):
        possibilityProgression = self.get_optimal_possibility_progression()
        return self.generate_realization_from_possibility_progression(possibilityProgression)


class FiguredBassLineException(exceptions21.Music21Exception):
    pass


class Test(unittest.TestCase):
    def testMultipleFiguresInLyric(self):
        from music21 import converter

        s = converter.parse('tinynotation: 4/4 C4 F4 G4_64 G4 C1', makeNotation=False)
        third_note = s[note.Note][2]
        self.assertEqual(third_note.lyric, '64')
        unused_fb = figured_bass_from_stream(s)
        self.assertEqual(third_note.editorial.notationString, '6, 4')

        third_note.lyric = '#6#42'
        unused_fb = figured_bass_from_stream(s)
        self.assertEqual(third_note.editorial.notationString, '#6, #4, 2')

        third_note.lyric = '#64#2'
        unused_fb = figured_bass_from_stream(s)
        self.assertEqual(third_note.editorial.notationString, '#6, 4, #2')

        # original case
        third_note.lyric = '6\n4'
        unused_fb = figured_bass_from_stream(s)
        self.assertEqual(third_note.editorial.notationString, '6, 4')

        # single accidental
        for single_symbol in '+#bn':
            with self.subTest(single_symbol=single_symbol):
                third_note.lyric = single_symbol
                unused_fb = figured_bass_from_stream(s)
                self.assertEqual(third_note.editorial.notationString, single_symbol)


if __name__ == '__main__':
    import music21

    music21.mainTest(Test)
