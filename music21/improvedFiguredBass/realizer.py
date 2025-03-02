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
from collections import defaultdict

from tqdm import tqdm

from music21 import chord, stream
from music21 import clef
from music21 import exceptions21
from music21 import key
from music21 import meter
from music21 import note
from music21 import pitch
from music21.figuredBass import checker
from music21.improvedFiguredBass import notation
from music21.improvedFiguredBass import realizer_scale
from music21.improvedFiguredBass import segment
from music21.improvedFiguredBass.possibility import Possibility
from music21.improvedFiguredBass.rules import RuleSet
from music21.improvedFiguredBass.skip_rules import SkipDecision
from music21.note import Note


def figured_bass_from_stream(stream_part) -> FiguredBassLine:
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
    def __init__(self, in_key=None, in_time=None):
        if in_key is None:
            in_key = key.Key('C')
        if in_time is None:
            in_time = meter.TimeSignature('4/4')

        self.inKey = in_key
        self.inTime = in_time
        self._paddingLeft = 0.0
        self._overlaidParts = stream.Part()
        self._fbScale = realizer_scale.FiguredBassScale(in_key.pitchFromDegree(1), in_key.mode)
        self._fbList = []

    def add_element(self, bass_object: note.Note, notation_string=None):
        """
        Use this method to add (bassNote, notationString) pairs to the bass line. Elements
        are realized in the order they are added.


        >>> from music21.improvedFiguredBass import realizer
        >>> fbLine = realizer.FiguredBassLine(key.Key('B'), meter.TimeSignature('3/4'))
        >>> fbLine.add_element(note.Note('B2'))
        >>> fbLine.add_element(note.Note('C#3'), '6')
        >>> fbLine.add_element(note.Note('D#3'), '6')
        """
        bass_object.editorial.notationString = notation_string
        c = bass_object.classes
        if 'Note' not in c:
            raise FiguredBassLineException(
                f'Not a valid bassObject (only note.Note supported) was {bass_object!r}')
        self._fbList.append((bass_object, notation_string))  # a bass note, and a notationString
        add_lyrics_to_bass_note(bass_object, notation_string)

    def generate_bass_line(self):
        bassLine = stream.Part()
        bassLine.append(clef.BassClef())
        bassLine.append(key.KeySignature(self.inKey.sharps))
        bassLine.append(copy.deepcopy(self.inTime))

        for (bassNote, unused_notationString) in self._fbList:
            bassLine.append(bassNote)

        return bassLine

    def retrieve_segments(self, max_pitch=None):
        '''
        generates the segmentList from an fbList, including any overlaid Segments

        if maxPitch is None, uses pitch.Pitch('B5')
        '''
        if max_pitch is None:
            max_pitch = pitch.Pitch('B5')
        segmentList = []
        bassLine = self.generate_bass_line()
        if self._overlaidParts:
            self._overlaidParts.append(bassLine)
            currentMapping = checker.extractHarmonies(self._overlaidParts)
        else:
            currentMapping = checker.createOffsetMapping(bassLine)

        offsets = [
            [n.offset, n.offset + n.quarterLength]
            for n in bassLine.flatten().notes
        ]
        for bass_note, play_offset in zip(bassLine.notes, offsets):
            currentSegment = segment.Segment(bass_note, bass_note.editorial.notationString,
                                             self._fbScale,
                                             max_pitch, play_offsets=play_offset)
            segmentList.append(currentSegment)
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
        if max_pitch is None:
            max_pitch = pitch.Pitch('B5')

        segmentList = self.retrieve_segments(max_pitch)
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
            self.rule_set = RuleSet()
        if 'start_offset' in fb_line_outputs:
            self.start_offset = fb_line_outputs['start_offset']
        else:
            self.start_offset = 0
        self.keyboardStyleOutput = True

    def generate_dp_table(self):
        first_possibilities = self.segment_list[0].all_filtered_possibilities(self.rule_set)

        dp: list[dict[int, dict[Possibility, int | float]]] = [  # (possibility, cost) dicts for each segment index i
            {
                self.rule_set.MAX_ALLOWANCE:
                    {possib: self.segment_list[0].get_cost(self.rule_set, possib) for possib in first_possibilities}
            }
        ]

        for i in range(len(self.segment_list) - 1):
            segment_b = self.segment_list[i + 1]
            possibs_to = segment_b.all_filtered_possibilities(self.rule_set)
            dp_entry = defaultdict(lambda: defaultdict(lambda: float('inf')))
            for possib in tqdm(possibs_to, leave=False, desc=f"Segment {i + 1}/{len(self.segment_list)}"):
                for segment_a_idx in range(i, -1, -1):
                    num_skips = i - segment_a_idx
                    segment_a = self.segment_list[segment_a_idx]
                    skip_decision = self.rule_set.should_skip(segment_a)
                    if skip_decision == SkipDecision.SKIP:
                        continue
                    for prev_avail, possib_cost_dict in dp[segment_a_idx].items():
                        for prev_possib, prev_cost in possib_cost_dict.items():
                            local_cost_b = segment_b.get_cost(self.rule_set, possib)

                            transition_cost = self.rule_set.get_cost(prev_possib, segment_a, possib, segment_b)
                            new_cost = prev_cost + (num_skips+1)*(transition_cost + local_cost_b)

                            increase_allowance = i > 0 and i % self.rule_set.INCREASE_ALLOWANCE_INTERVAL == 0
                            avail = min(prev_avail + increase_allowance, self.rule_set.MAX_ALLOWANCE)
                            dp_entry[avail][possib] = min(dp_entry[avail][possib], new_cost)
                            if prev_avail >= 1:
                                for intermediate_pitch, voice in segment_a.get_intermediate_int_pitches(prev_possib, possib):
                                    transition_cost = self.rule_set.get_cost_with_intermediate(
                                        prev_possib, segment_a, possib, segment_b, intermediate_pitch, voice)
                                    new_cost = prev_cost + (num_skips + 1) * (transition_cost + local_cost_b)
                                    new_avail = prev_avail - 1
                                    dp_entry[new_avail][possib] = min(dp_entry[new_avail][possib], new_cost)
                    if skip_decision == SkipDecision.NO_SKIP:
                        break

            dp.append(dp_entry)
        return dp

    def get_reverse_choices(self, dp):
        # Extract best possib
        reverse_progression = []
        best_possib = None
        final_cost = float('inf')
        for _, d in dp[-1].items():
            new_cost = min(d.values())
            if new_cost < final_cost:
                final_cost = new_cost
                best_possib = min(d, key=d.get)
        best_cost = final_cost
        reverse_progression.append((best_possib, 0, None))

        logging.log(logging.INFO, f"======================================")
        logging.log(logging.INFO, f"Found solution with cost {final_cost}.")
        logging.log(logging.INFO, f"======================================")
        i = len(self.segment_list) - 2
        while i >= 0:
            segment_b = self.segment_list[i + 1]
            for segment_a_idx in range(i, -1, -1):
                num_skips = i - segment_a_idx
                segment_a = self.segment_list[segment_a_idx]
                found = False
                for prev_avail, possib_cost_dict in dp[segment_a_idx].items():
                    for possib_a, prev_cost in possib_cost_dict.items():
                        to_possib_cost = segment_b.get_cost(self.rule_set, best_possib)
                        transition_cost = self.rule_set.get_cost(possib_a, segment_a, best_possib, segment_b)
                        if (num_skips+1) * (transition_cost + to_possib_cost) == best_cost - prev_cost:
                            best_possib = possib_a
                            best_cost = prev_cost
                            reverse_progression.append((best_possib, num_skips, None))
                            logging.log(logging.INFO, f"Chose {best_possib} after skipping {num_skips} notes.")
                            i -= num_skips
                            found = True
                            break
                        if prev_avail >= 1:
                            for intermediate_pitch, voice in segment_a.get_intermediate_int_pitches(possib_a, best_possib):
                                transition_cost = self.rule_set.get_cost_with_intermediate(
                                    possib_a, segment_a, best_possib, segment_b, intermediate_pitch, voice)
                                if (num_skips+1) * (transition_cost + to_possib_cost) == best_cost - prev_cost:
                                    best_possib = possib_a
                                    best_cost = prev_cost
                                    reverse_progression.append((best_possib, num_skips, intermediate_pitch))
                                    logging.log(logging.INFO, f"Chose {best_possib} after skipping {num_skips} notes"
                                                              f" with intermediate note {intermediate_pitch}.")
                                    i -= num_skips
                                    found = True
                                    break
                            if found:
                                break
                    if found:
                        break
                if found:
                    break
            i -= 1
        return reverse_progression

    def get_optimal_possibility_progression(self):
        """
        Returns a random unique possibility progression.

        Mutates segment_list
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
        for val, num_skips, intermediate_note in reversed(reverse_progression):
            curr_segment = self.segment_list[curr_idx]
            curr_segment.intermediate_note = intermediate_note
            m = curr_segment.measure_number
            if m is not None and m < measure:
                measure = m
                logging.log(logging.INFO, f"### Measure {measure} ###")
            if prev_val:
                self.rule_set.get_cost(val, curr_segment, prev_val, prev_segment, enable_logging=True)

            self.rule_set.get_cost(val, curr_segment, enable_logging=True)
            result.append(val)
            for i in range(curr_idx+1, curr_idx+1+num_skips):
                self.segment_list[curr_idx].duration.quarterLength += self.segment_list[i].quarterLength
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

        if self.keyboardStyleOutput:
            rightHand = stream.Part()
            sol.insert(0.0, rightHand)
            rightHand.append([copy.deepcopy(self._keySig), copy.deepcopy(self._inTime)])

            for segmentIndex in range(len(self.segment_list)):
                possibA = possibility_progression[segmentIndex].get_pitches()
                for p in possibA:
                    if p.accidental is not None:
                        p.accidental.displayStatus = False
                bassNote = copy.deepcopy(self.segment_list[segmentIndex].bassNote)
                bassLine.append(bassNote)
                rhPitches = possibA[0:-1]
                rhChord = chord.Chord(rhPitches)
                rhChord.quarterLength = self.segment_list[segmentIndex].quarterLength
                intermediate_note = self.segment_list[segmentIndex].intermediate_note
                rightHand.append(rhChord)
                if intermediate_note:
                    rhChord.quarterLength /= 2
                    n = Note(intermediate_note)
                    n.pitch.accidental.displayStatus = False
                    n.quarterLength = rhChord.quarterLength
                    rightHand.append(n)
            rightHand.insert(0.0, clef.TrebleClef())

        else:  # Chorale-style output
            upperParts = []
            for partNumber in range(len(possibility_progression[0].pitches) - 1):
                fbPart = stream.Part()
                sol.insert(0.0, fbPart)
                fbPart.append([copy.deepcopy(self._keySig), copy.deepcopy(self._inTime)])
                upperParts.append(fbPart)

            for segmentIndex in range(len(self.segment_list)):
                possibA = possibility_progression[segmentIndex].pitches
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

        bassLine.insert(0.0, clef.BassClef())
        bassLine.makeNotation(inPlace=True, cautionaryNotImmediateRepeat=False)
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
