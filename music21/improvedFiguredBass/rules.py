import itertools
import logging
import math
import unittest
from abc import abstractmethod, ABC
from functools import cache

from music21 import voiceLeading, pitch
from music21.improvedFiguredBass.rules_config import RulesConfig


class RuleSet:
    MAX_SINGLE_POSSIB_COST = 10000

    def __init__(self, conf: RulesConfig):
        self.config = conf

        self.rules = [
            ParallelFifths(cost=conf.highPriorityRuleCost),
            ParallelOctaves(cost=conf.highPriorityRuleCost),
            HiddenFifth(cost=conf.lowPriorityRuleCost),
            HiddenOctave(cost=conf.lowPriorityRuleCost),
            VoiceOverlap(cost=conf.highPriorityRuleCost),
            #             # UpperPartsSame(cost=conf.lowPriorityRuleCost),
            MinimizeMovementsMiddleVoices(cost=conf.lowPriorityRuleCost),
            MinimizeMovementsSopranoVoice(cost=conf.highPriorityRuleCost),
            UnpreparedNote(cost=conf.lowPriorityRuleCost),
        ]

        self.single_rules = [
            VoiceCrossing(cost=float('inf')),
            HasDuplicate(cost=float('inf')),
            LimitPartToPitch(cost=5),
            NoSecondInterval(cost=float('inf')),
            IsIncomplete(cost=6),
            UpperPartsWithinLimit(cost=2 * conf.highPriorityRuleCost),
            IsPlayable(cost=float('inf')),
            PitchesWithinLimit(cost=float('inf')),
            AvoidSeventhChord(cost=7),
        ]

    def get_rules(self):
        return self.rules

    def get_cost(self, possib_a, possib_b=None, context=None, enable_logging=False):
        total_cost = 0
        rules: list[Rule | SingleRule] = self.rules if possib_b is not None else self.single_rules
        for rule in rules:
            if possib_b is not None:
                cost = rule.get_cost(possib_a, possib_b, context)
            else:
                cost = rule.get_cost(possib_a, context)
            if enable_logging and cost > 0:
                logging.log(logging.INFO, f"Cost += {cost} due to {rule.__class__.__name__}")
            total_cost += cost
            if total_cost == float('inf'): return total_cost
        return total_cost


class Rule(ABC):
    def __init__(self, cost):
        self.cost = cost

    @abstractmethod
    def get_cost(self, possib_a, possib_b, context):
        pass

    def get_all_pair_possibs(self, possib_a, possib_b):
        if len(possib_a) > len(possib_b):
            return self.get_all_pair_possibs(possib_b, possib_a)
        elif len(possib_a) == len(possib_b):
            return [list(zip(possib_a, possib_a))]

        result = []
        n = len(possib_a)
        m = len(possib_b)
        for to_tuple in itertools.combinations(range(m), n):
            from_tuple = tuple(range(n))
            note_pairs = [(possib_a[f], possib_b[t]) for f, t in zip(from_tuple, to_tuple)]
            result.append(tuple(note_pairs))
        return result


class ParallelFifths(Rule):

    def get_cost(self, possib_a, possib_b, _):
        return self.cost if self.hasParallelFifths(possib_a, possib_b) else 0

    @cache
    def hasParallelFifths(self, possibA, possibB):
        hasParallelFifths = False
        pairsList = partPairs(possibA, possibB)

        for pair1Index in range(len(pairsList)):
            (higherPitchA, higherPitchB) = pairsList[pair1Index]
            for pair2Index in range(pair1Index + 1, len(pairsList)):
                (lowerPitchA, lowerPitchB) = pairsList[pair2Index]
                if not abs(higherPitchA.ps - lowerPitchA.ps) % 12 == 7:
                    continue
                if not abs(higherPitchB.ps - lowerPitchB.ps) % 12 == 7:
                    continue
                # Very high probability of ||5, but still not certain.
                pitchQuartet = (lowerPitchA, lowerPitchB, higherPitchA, higherPitchB)
                vlq = voiceLeading.VoiceLeadingQuartet(*pitchQuartet)
                if vlq.parallelFifth():
                    hasParallelFifths = True
                if hasParallelFifths:
                    return hasParallelFifths

        return hasParallelFifths


class ParallelOctaves(Rule):
    def get_cost(self, possib_a, possib_b, _):
        return self.cost if self.has_parallel_octaves(possib_a, possib_b) else 0

    @cache
    def has_parallel_octaves(self, possibA, possibB):
        '''
        Returns True if there are parallel octaves between any
        two shared parts of possibA and possibB.


        If pitchA1 and pitchA2 in possibA are separated by
        a simple interval of a perfect octave, and they move
        to a pitchB1 and pitchB2 in possibB also separated
        by the simple interval of a perfect octave, then this
        constitutes parallel octaves between these two parts.


        If the method returns False, then no two shared parts
        have parallel octaves. The method returns True as soon
        as two shared parts with parallel octaves are found.

        >>> from music21.figuredBass import possibility
        >>> C3 = pitch.Pitch('C3')
        >>> D3 = pitch.Pitch('D3')
        >>> G3 = pitch.Pitch('G3')
        >>> A3 = pitch.Pitch('A3')
        >>> C4 = pitch.Pitch('C4')
        >>> D4 = pitch.Pitch('D4')


        Here, the soprano moves from C4 to D4 and the bass moves
        from C3 to D3. The interval between C3 and C4, as well as
        between D3 and D4, is a parallel octave. The two parts,
        and therefore the two possibilities, have parallel octaves.


        >>> possibA1 = (C4, G3, C3)
        >>> possibB1 = (D4, A3, D3)
        >>> possibility.parallelOctaves(possibA1, possibB1)
        True


        Now, the soprano moves down to B3. The interval between
        D3 and B3 is a major sixth. The soprano and bass parts
        no longer have parallel octaves. The tenor part forms
        a parallel octave with neither the bass nor soprano,
        so the two possibilities do not have parallel octaves.
        (Notice, however, the parallel fifth between the bass
        and tenor!)


        >>> B3 = pitch.Pitch('B3')
        >>> possibA2 = (C4, G3, C3)
        >>> possibB2 = (B3, A3, D3)
        >>> possibility.parallelOctaves(possibA2, possibB2)
        False
        '''
        hasParallelOctaves = False
        pairsList = partPairs(possibA, possibB)

        for pair1Index in range(len(pairsList)):
            (higherPitchA, higherPitchB) = pairsList[pair1Index]
            for pair2Index in range(pair1Index + 1, len(pairsList)):
                (lowerPitchA, lowerPitchB) = pairsList[pair2Index]
                if not abs(higherPitchA.ps - lowerPitchA.ps) % 12 == 0:
                    continue
                if not abs(higherPitchB.ps - lowerPitchB.ps) % 12 == 0:
                    continue
                # Very high probability of ||8, but still not certain.
                pitchQuartet = (lowerPitchA, lowerPitchB, higherPitchA, higherPitchB)
                vlq = voiceLeading.VoiceLeadingQuartet(*pitchQuartet)
                if vlq.parallelOctave():
                    hasParallelOctaves = True
                if hasParallelOctaves:
                    return hasParallelOctaves

        return hasParallelOctaves


class HiddenFifth(Rule):

    def get_cost(self, possib_a, possib_b, _):
        return self.cost if self.has_hidden_fifth(possib_a, possib_b) else 0

    @cache
    def has_hidden_fifth(self, possibA, possibB):
        '''
        Returns True if there is a hidden fifth between shared outer parts
        of possibA and possibB. The outer parts here are the first and last
        elements of each possibility.


        If sopranoPitchA and bassPitchA in possibA move to a sopranoPitchB
        and bassPitchB in possibB in similar motion, and the simple interval
        between sopranoPitchB and bassPitchB is that of a perfect fifth,
        then this constitutes a hidden octave between the two possibilities.

        >>> from music21.figuredBass import possibility
        >>> C3 = pitch.Pitch('C3')
        >>> D3 = pitch.Pitch('D3')
        >>> E3 = pitch.Pitch('E3')
        >>> F3 = pitch.Pitch('F3')
        >>> E5 = pitch.Pitch('E5')
        >>> A5 = pitch.Pitch('A5')


        Here, the bass part moves up from C3 to D3 and the soprano part moves
        up from E5 to A5. The simple interval between D3 and A5 is a perfect
        fifth. Therefore, there is a hidden fifth between the two possibilities.


        >>> possibA1 = (E5, E3, C3)
        >>> possibB1 = (A5, F3, D3)
        >>> possibility.hiddenFifth(possibA1, possibB1)
        True


        Here, the soprano and bass parts also move in similar motion, but the
        simple interval between D3 and Ab5 is a diminished fifth. Consequently,
        there is no hidden fifth.


        >>> Ab5 = pitch.Pitch('A-5')
        >>> possibA2 = (E5, E3, C3)
        >>> possibB2 = (Ab5, F3, D3)
        >>> possibility.hiddenFifth(possibA2, possibB2)
        False


        Now, we have the soprano and bass parts again moving to A5 and D3, whose
        simple interval is a perfect fifth. However, the bass moves up while the
        soprano moves down. Therefore, there is no hidden fifth.


        >>> E6 = pitch.Pitch('E6')
        >>> possibA3 = (E6, E3, C3)
        >>> possibB3 = (A5, F3, D3)
        >>> possibility.hiddenFifth(possibA3, possibB3)
        False
        '''
        hasHiddenFifth = False
        pairsList = partPairs(possibA, possibB)
        (highestPitchA, highestPitchB) = pairsList[0]
        (lowestPitchA, lowestPitchB) = pairsList[-1]

        if abs(highestPitchB.ps - lowestPitchB.ps) % 12 == 7:
            # Very high probability of hidden fifth, but still not certain.
            pitchQuartet = (lowestPitchA, lowestPitchB, highestPitchA, highestPitchB)
            vlq = voiceLeading.VoiceLeadingQuartet(*pitchQuartet)
            if vlq.hiddenFifth():
                hasHiddenFifth = True

        return hasHiddenFifth


class HiddenOctave(Rule):
    def get_cost(self, possib_a, possib_b, _):
        return self.cost if self.has_hidden_octave(possib_a, possib_b) else 0

    @cache
    def has_hidden_octave(self, possibA, possibB):
        '''
        Returns True if there is a hidden octave between shared outer parts
        of possibA and possibB. The outer parts here are the first and last
        elements of each possibility.


        If sopranoPitchA and bassPitchA in possibA move to a sopranoPitchB
        and bassPitchB in possibB in similar motion, and the simple interval
        between sopranoPitchB and bassPitchB is that of a perfect octave,
        then this constitutes a hidden octave between the two possibilities.

        >>> from music21.figuredBass import possibility
        >>> C3 = pitch.Pitch('C3')
        >>> D3 = pitch.Pitch('D3')
        >>> E3 = pitch.Pitch('E3')
        >>> F3 = pitch.Pitch('F3')
        >>> A5 = pitch.Pitch('A5')
        >>> D6 = pitch.Pitch('D6')


        Here, the bass part moves up from C3 to D3 and the soprano part moves
        up from A5 to D6. The simple interval between D3 and D6 is a perfect
        octave. Therefore, there is a hidden octave between the two possibilities.


        >>> possibA1 = (A5, E3, C3)
        >>> possibB1 = (D6, F3, D3)  # Perfect octave between soprano and bass.
        >>> possibility.hiddenOctave(possibA1, possibB1)
        True


        Here, the bass part moves up from C3 to D3 but the soprano part moves
        down from A6 to D6. There is no hidden octave since the parts move in
        contrary motion.


        >>> A6 = pitch.Pitch('A6')
        >>> possibA2 = (A6, E3, C3)
        >>> possibB2 = (D6, F3, D3)
        >>> possibility.hiddenOctave(possibA2, possibB2)
        False
        '''
        hasHiddenOctave = False
        pairsList = partPairs(possibA, possibB)
        (highestPitchA, highestPitchB) = pairsList[0]
        (lowestPitchA, lowestPitchB) = pairsList[-1]

        if abs(highestPitchB.ps - lowestPitchB.ps) % 12 == 0:
            # Very high probability of hidden octave, but still not certain.
            pitchQuartet = (lowestPitchA, lowestPitchB, highestPitchA, highestPitchB)
            vlq = voiceLeading.VoiceLeadingQuartet(*pitchQuartet)
            if vlq.hiddenOctave():
                hasHiddenOctave = True

        return hasHiddenOctave


class VoiceOverlap(Rule):
    def get_cost(self, possib_a, possib_b, _):
        return self.cost if self.has_voice_overlap(possib_a, possib_b) else 0

    @cache
    def has_voice_overlap(self, possib_a, possib_b):
        for pairs in self.get_all_pair_possibs(possib_a[1:-1], possib_b[1:-1]):
            all_pairs = [(possib_a[0], possib_b[0])] + list(pairs) + [(possib_a[-1], possib_b[-1])]
            possibs = list(zip(*all_pairs))
            if not self.has_voice_overlap_equal(possibs[0], possibs[1]):
                return False
        return True

    @cache
    def has_voice_overlap_equal(self, possibA, possibB):
        '''
        Returns True if there is voice overlap between any two shared parts
        of possibA and possibB.


        Voice overlap can occur in two ways:


        1) If a pitch in a lower part in possibB is higher than a pitch in
        a higher part in possibA. This case is demonstrated below.


        2) If a pitch in a higher part in possibB is lower than a pitch in
        a lower part in possibA.


            .. image:: images/figuredBass/fbPossib_voiceOverlap.*
                :width: 75


        In the above example, possibA has G4 in the bass and B4 in the soprano.
        If the bass moves up to C5 in possibB, that would constitute voice overlap
        because the bass in possibB would be higher than the soprano in possibA.

        >>> from music21.figuredBass import possibility
        >>> C4 = pitch.Pitch('C4')
        >>> D4 = pitch.Pitch('D4')
        >>> E4 = pitch.Pitch('E4')
        >>> F4 = pitch.Pitch('F4')
        >>> G4 = pitch.Pitch('G4')
        >>> C5 = pitch.Pitch('C5')


        Here, case #2 is demonstrated. There is overlap between the soprano and
        alto parts, because F4 in the soprano in possibB1 is lower than the G4
        in the alto in possibA1. Note that neither possibility has to have voice
        crossing for voice overlap to occur, as shown.


        >>> possibA1 = (C5, G4, E4, C4)
        >>> possibB1 = (F4, F4, D4, D4)
        >>> possibility.voiceOverlap(possibA1, possibB1)
        True
        >>> possibility.voiceCrossing(possibA1)
        False
        >>> possibility.voiceCrossing(possibB1)
        False


        Here is the same example as above, except the soprano of the second
        possibility is now B4, which does not overlap the G4 of the first.
        Now, there is no voice overlap.


        >>> B4 = pitch.Pitch('B4')
        >>> possibA2 = (C5, G4, E4, C4)
        >>> possibB2 = (B4, F4, D4, D4)
        >>> possibility.voiceOverlap(possibA2, possibB2)
        False
        '''
        hasVoiceOverlap = False
        pairsList = partPairs(possibA, possibB)

        for pair1Index in range(len(pairsList)):
            (higherPitchA, higherPitchB) = pairsList[pair1Index]
            for pair2Index in range(pair1Index + 1, len(pairsList)):
                (lowerPitchA, lowerPitchB) = pairsList[pair2Index]
                if lowerPitchB > higherPitchA or higherPitchB < lowerPitchA:
                    hasVoiceOverlap = True
                    return hasVoiceOverlap

        return hasVoiceOverlap


class MinimizeMovementsMiddleVoices(Rule):
    def get_cost(self, possib_a, possib_b, context=None):
        diff = self.get_minimum_difference(possib_a, possib_b)
        if diff == 0 and self.cost == float('inf'): return 0
        return self.cost * diff / max(len(possib_a), len(possib_b))

    def get_minimum_difference(self, possib_a, possib_b):
        """
        >>> from music21.improvedFiguredBass.rules import MinimizeMovementsMiddleVoices
        >>> from music21.pitch import Pitch
        >>> a = (Pitch('C5'), Pitch('G4'), Pitch('E4'), Pitch('C3'))
        >>> b = (Pitch('C5'), Pitch('A4'), Pitch('E4'), Pitch('C4'), Pitch('E3'), Pitch('A2'))
        >>> m = MinimizeMovementsMiddleVoices(cost=1)
        >>> m.get_cost(a, b)
        5
        >>> a = (Pitch('A4'))
        """
        middle_left = self.get_all_pair_possibs(possib_a[1:-1], possib_b[1:-1])
        possibs = []
        min_diff = float('inf')
        for possib in middle_left:
            diff = 0
            for a, b in possib:
                diff += self.distance_between(a, b)
            if diff < min_diff:
                possibs = possib
                min_diff = diff
        possibs = list(possibs)
        possibs.append((possib_a[0], possib_b[0]))
        possibs.append((possib_a[-1], possib_b[-1]))
        not_matched = set(possib_b)
        for a, b in possibs:
            if b in not_matched:
                not_matched.remove(b)
        for b in not_matched:
            closest = None
            dist = float('inf')
            for p in possib_a:
                if (w := self.distance_between(p, b)) < dist:
                    closest = p
                    dist = w
            min_diff += dist
            possibs.append((closest, b))
        return min_diff

    @staticmethod
    def distance_between(part_a, part_b):
        return math.ceil(max(abs(part_a.ps - part_b.ps) / 2 - 1, 0))


class MinimizeMovementsSopranoVoice(Rule):
    def get_cost(self, possib_a, possib_b, context):
        diff = self.distance_between(possib_a[0], possib_b[0])
        if diff == 0 and self.cost == float('inf'):
            return 0
        else:
            return math.floor(diff / 2) * self.cost

    @staticmethod
    def distance_between(part_a, part_b):
        return abs(part_a.ps - part_b.ps)


class PartMovementsWithinLimits(Rule):
    def __init__(self, cost, limits):
        super().__init__(cost)
        self.limits = limits

    def get_cost(self, possib_a, possib_b, _):
        return self.cost if not self.part_movements_within_limits(possib_a, possib_b) else 0

    @cache
    def part_movements_within_limits(self, possibA, possibB):
        # noinspection PyShadowingNames
        '''
        Returns True if all movements between shared parts of possibA and possibB
        are within limits, as specified by list partMovementLimits, which consists of
        (partNumber, maxSeparation) tuples.

        * partNumber: Specified from 1 to n, where 1 is the soprano or
          highest part and n is the bass or lowest part.

        * maxSeparation: For a given part, the maximum separation to allow
          between a pitch in possibA and a corresponding pitch in possibB, in semitones.

        >>> from music21.figuredBass import possibility
        >>> C4 = pitch.Pitch('C4')
        >>> D4 = pitch.Pitch('D4')
        >>> E4 = pitch.Pitch('E4')
        >>> F4 = pitch.Pitch('F4')
        >>> G4 = pitch.Pitch('G4')
        >>> A4 = pitch.Pitch('A4')
        >>> B4 = pitch.Pitch('B4')
        >>> C5 = pitch.Pitch('C5')

        Here, we limit the soprano part to motion of two semitones,
        enharmonically equivalent to a major second.
        Moving from C5 to B4 is allowed because it constitutes stepwise
        motion, but moving to A4 is not allowed
        because the distance between A4 and C5 is three semitones.

        >>> partMovementLimits = [(1, 2)]
        >>> possibA1 = (C5, G4, E4, C4)
        >>> possibB1 = (B4, F4, D4, D4)
        >>> possibility.partMovementsWithinLimits(possibA1, possibB1, partMovementLimits)
        True
        >>> possibB2 = (A4, F4, D4, D4)
        >>> possibility.partMovementsWithinLimits(possibA1, possibB2, partMovementLimits)
        False
        '''
        withinLimits = True
        for (partNumber, maxSeparation) in self.limits:
            pitchA = possibA[partNumber - 1]
            pitchB = possibB[partNumber - 1]
            if abs(pitchB.ps - pitchA.ps) > maxSeparation:
                withinLimits = False
                return withinLimits

        return withinLimits


class UpperPartsSame(Rule):
    def get_cost(self, possib_a, possib_b, _):
        return self.cost if not self.upper_parts_same(possib_a, possib_b) else 0

    @cache
    def upper_parts_same(self, possibA, possibB):
        '''
        Returns True if the upper parts are the same.
        False otherwise.

        >>> from music21.figuredBass import possibility
        >>> C4 = pitch.Pitch('C4')
        >>> D4 = pitch.Pitch('D4')
        >>> E4 = pitch.Pitch('E4')
        >>> F4 = pitch.Pitch('F4')
        >>> G4 = pitch.Pitch('G4')
        >>> B4 = pitch.Pitch('B4')
        >>> C5 = pitch.Pitch('C5')
        >>> possibA1 = (C5, G4, E4, C4)
        >>> possibB1 = (B4, F4, D4, D4)
        >>> possibB2 = (C5, G4, E4, D4)
        >>> possibility.upperPartsSame(possibA1, possibB1)
        False
        >>> possibility.upperPartsSame(possibA1, possibB2)
        True
        '''
        pairsList = partPairs(possibA, possibB)

        for (pitchA, pitchB) in pairsList[0:-1]:
            if not (pitchA == pitchB):
                return False

        return True


class PartsSame(Rule):
    def __init__(self, cost, parts_to_check):
        super().__init__(cost)
        self.parts_to_check = parts_to_check

    def get_cost(self, possib_a, possib_b, _):
        return self.cost if not self.are_parts_same(possib_a, possib_b) else 0

    @cache
    def are_parts_same(self, possib_a, possib_b):
        '''
        Takes in partsToCheck, a list of part numbers. Checks if pitches at those part numbers of
        possibA and possibB are equal, determined by pitch space.

        >>> from music21.figuredBass import possibility
        >>> C4 = pitch.Pitch('C4')
        >>> E4 = pitch.Pitch('E4')
        >>> G4 = pitch.Pitch('G4')
        >>> B4 = pitch.Pitch('B4')
        >>> C5 = pitch.Pitch('C5')
        >>> possibA1 = (C5, G4, E4, C4)
        >>> possibB1 = (B4, G4, E4, C4)
        >>> possibility.partsSame(possibA1, possibB1, [2, 3, 4])
        True
        '''
        if self.parts_to_check is None:
            return True

        pairsList = partPairs(possib_a, possib_b)

        for partIndex in self.parts_to_check:
            (pitchA, pitchB) = pairsList[partIndex - 1]
            if pitchA != pitchB:
                return False

        return True


class UnpreparedNote(Rule):
    def get_cost(self, possib_a, possib_b, context):
        segment_b = context['segment_b']
        return self.cost if self.has_unprepared_note(possib_a, possib_b, segment_b) else 0

    @cache
    def has_unprepared_note(self, possibA, possibB, segmentB):
        '''
        >>> from music21.improvedFiguredBass import segment
        >>> C = pitch.Pitch("C3")
        >>> E = pitch.Pitch("E3")
        >>> F = pitch.Pitch("F3")
        >>> G = pitch.Pitch("G3")
        >>> A = pitch.Pitch("A3")
        >>> B = pitch.Pitch("B3")
        >>> Ch = pitch.Pitch("C4")
        >>> Eh = pitch.Pitch("E4")
        >>> Fh = pitch.Pitch("F4")
        >>> possibAPrepared = (Eh, B, G, E)
        >>> possibAUnprepared = (Fh, Ch, A, F)
        >>> possibB = (B, G, E, C)
        >>> segmentB = segment.Segment(notationString='7')
        >>> hasUnpreparedNote(possibAUnprepared, possibB, segmentB)
        True
        >>> hasUnpreparedNote(possibAPrepared, possibB, segmentB)
        False
        '''
        seventh = None
        ninth = None
        for segmentChord in segmentB.segmentChord:
            pos = segmentChord.getChordStep(7)
            if pos:
                seventh = pos
            pos = segmentChord.getChordStep(7)
            if pos:
                ninth = pos
        if seventh is not None:
            for n2 in possibB:
                if n2.pitchClass == seventh.pitchClass and n2 not in possibA:
                    return True

        if ninth is not None:
            for n2 in possibB:
                if n2.pitchClass == ninth.pitchClass and n2 not in possibA:
                    return True
        return False


class SingleRule(ABC):
    def __init__(self, cost):
        self.cost = cost

    @abstractmethod
    def get_cost(self, possib_a, context):
        pass


class AvoidSeventhChord(SingleRule):
    def get_cost(self, possib_a, context):
        segment = context['segment']
        if len(segment.segmentChord) <= 1:
            return 0
        classes = [p.pitchClass for p in segment.segmentChord[0].pitches]
        for pitch in possib_a:
            if pitch.pitchClass not in classes:
                return self.cost
        return 0


class IsPlayable(SingleRule):
    def get_cost(self, possib_a, context):
        return self.cost if not self.is_playable(possib_a) else 0

    def is_playable(self, possib_a):
        if len(possib_a) < 5:
            return self.playable_by_one_hand(possib_a[:-1])
        else:
            return self.playable_by_one_hand(possib_a[:-2]) and self.playable_by_one_hand(possib_a[-2:])

    @staticmethod
    def playable_by_one_hand(notes):
        return notes[0].ps - notes[-1].ps <= 12


class NoSecondInterval(SingleRule):
    def get_cost(self, possib_a, context):
        for i in range(len(possib_a) - 1):
            p1 = possib_a[i]
            p2 = possib_a[i + 1]
            if p1.ps - p2.ps <= 2:
                return self.cost
        return 0


class VoiceCrossing(SingleRule):
    def get_cost(self, possib_a, context):
        return self.cost if self.voiceCrossing(possib_a) else 0

    def voiceCrossing(self, possibA):
        for part1Index in range(len(possibA)):
            higherPitch = possibA[part1Index]
            for part2Index in range(part1Index + 1, len(possibA)):
                lowerPitch = possibA[part2Index]
                if higherPitch < lowerPitch:
                    return True

        return False


class HasDuplicate(SingleRule):
    def get_cost(self, possib_a, context):
        return self.cost if len(possib_a) != len(set(possib_a)) else 0


class IsIncomplete(SingleRule):
    def get_cost(self, possib_a, context):
        needed_pitch_names = context['segment'].pitchNamesInChord
        melody_notes = context['segment'].melody_notes
        for pitch_names in needed_pitch_names:
            if not self.isIncomplete(possib_a, pitch_names.copy(), melody_notes):
                return 0
        return self.cost

    def isIncomplete(self, possibA, pitchNamesToContain, melody_notes):
        for note in melody_notes:
            if note in pitchNamesToContain:
                pitchNamesToContain.remove(note)
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


class UpperPartsWithinLimit(SingleRule):
    def get_cost(self, possib_a, context):
        return self.cost if not self.upperPartsWithinLimit(possib_a) else 0

    def upperPartsWithinLimit(self, possibA, maxSemitoneSeparation=12):
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

        if len(possibA) < 3:
            return True
        return possibA[0].ps - possibA[-2].ps <= maxSemitoneSeparation


class PitchesWithinLimit(SingleRule):
    def get_cost(self, possib_a, context):
        return self.cost if not self.pitchesWithinLimit(possib_a) else 0

    def pitchesWithinLimit(self, possibA, maxPitch=pitch.Pitch('B5'), minRightHandPitch=pitch.Pitch('A3')):
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
        if possibA[0] > maxPitch:
            return False
        if len(possibA) > 4:
            lowestRightHandNote = possibA[-3]
        else:
            lowestRightHandNote = possibA[-2]
        if lowestRightHandNote < minRightHandPitch:
            return False

        return True


class LimitPartToPitch(SingleRule):
    def get_cost(self, possib_a, context):
        return self.cost if not self.limitPartToPitch(possib_a) else 0

    def limitPartToPitch(self, possibA, partPitchLimits=None):
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


# HELPER METHODS
# --------------


def partPairs(possibA, possibB):
    '''
    Groups together pitches of possibA and possibB which correspond to the same part,
    constituting a shared part.

    >>> from music21.figuredBass import possibility
    >>> C4 = pitch.Pitch('C4')
    >>> D4 = pitch.Pitch('D4')
    >>> E4 = pitch.Pitch('E4')
    >>> F4 = pitch.Pitch('F4')
    >>> G4 = pitch.Pitch('G4')
    >>> B4 = pitch.Pitch('B4')
    >>> C5 = pitch.Pitch('C5')
    >>> possibA1 = (C5, G4, E4, C4)
    >>> possibB1 = (B4, F4, D4, D4)
    >>> possibility.partPairs(possibA1, possibA1)
    [(<music21.pitch.Pitch C5>, <music21.pitch.Pitch C5>), (<music21.pitch.Pitch G4>, <music21.pitch.Pitch G4>), (<music21.pitch.Pitch E4>, <music21.pitch.Pitch E4>), (<music21.pitch.Pitch C4>, <music21.pitch.Pitch C4>)]
    >>> possibility.partPairs(possibA1, possibB1)
    [(<music21.pitch.Pitch C5>, <music21.pitch.Pitch B4>), (<music21.pitch.Pitch G4>, <music21.pitch.Pitch F4>), (<music21.pitch.Pitch E4>, <music21.pitch.Pitch D4>), (<music21.pitch.Pitch C4>, <music21.pitch.Pitch D4>)]

    '''
    return list(zip(possibA, possibB))


class Test(unittest.TestCase):
    pass


if __name__ == '__main__':
    import music21

    music21.mainTest(Test)
