import itertools
import logging
import math

from abc import abstractmethod, ABC
from functools import cache
from typing import TYPE_CHECKING

from music21.improvedFiguredBass.possibility import Possibility
from music21.improvedFiguredBass.skip_rules import SkipDecision, SkipRules
from music21.pitch import Pitch

if TYPE_CHECKING:
    from music21.improvedFiguredBass.segment import Segment


class RuleSet:
    MAX_SINGLE_POSSIB_COST = 10e10

    # number of parts range for each dynamic marking
    DYNAMIC_RANGES = {
        'ppp': [2, 3],
        'pp': [2, 3],
        'p': [3, 3],
        'mp': [3, 4],
        'mf': [3, 4],
        'f': [4, 4],
        'ff': [5, 6],
        'fff': [5, 6]
    }

    MAX_COST = 10e4
    HIGH_COST = 1600
    MEDIUM_COST = 800
    LOW_COST = 400

    INCREASE_ALLOWANCE_INTERVAL = 40
    MAX_ALLOWANCE = 1

    def __init__(self):
        self.transition_rules = [
            ParallelFifths(cost=2*self.HIGH_COST),
            ParallelOctaves(cost=2*self.HIGH_COST),
            HiddenFifth(cost=self.LOW_COST),
            HiddenOctave(cost=self.LOW_COST),
            VoiceOverlap(cost=self.HIGH_COST),
            # UpperPartsSame(cost=self.lowPriorityRuleCost),
            MinimizeMovementsMiddleVoices(cost=0.5*self.LOW_COST),
            MinimizeMovementsSopranoVoice(cost=self.MEDIUM_COST),
            UnpreparedNote(cost=self.LOW_COST),
            CounterMovement(cost=self.MEDIUM_COST)
        ]

        self.single_rules = [
            # VoiceCrossing(cost=float('inf')),
            NotesFromFigures(cost=float('inf')),
            ContainRoot(cost=float('inf')),
            HasDuplicate(cost=float('inf')),
            NoSmallSecondInterval(cost=float('inf')),
            IsPlayable(cost=float('inf')),
            PitchesWithinLimit(cost=float('inf')),
            OnlyAllowSixOptionAfterCadence(cost=float('inf')),
            DoubleRootIfCadence(cost=self.HIGH_COST),
            UpperPartsWithinLimit(cost=self.HIGH_COST),
            AvoidDoubling(cost=self.MEDIUM_COST),
            IsIncomplete(cost=self.MEDIUM_COST),
            NotTooLow(cost=self.MEDIUM_COST),
            # AvoidSeventhChord(cost=self.LOW_COST),
            UseLeastAmountOfNotes(cost=self.MEDIUM_COST, dynamic_ranges=self.DYNAMIC_RANGES),
            PitchesUnderMelody(cost=0.5 * self.LOW_COST),
        ]

        self.skip_rules = SkipRules()

    @cache
    def get_cost_with_intermediate(
        self,
        possib_a: Possibility, segment_a: 'Segment',
        possib_b: Possibility, segment_b: 'Segment',
        intermediate_pitch: Pitch, voice=0, enable_logging=False
    ):
        new_pitches = list(possib_a.integer_pitches)
        new_pitches[voice] = intermediate_pitch
        new_possib_a = Possibility(tuple(new_pitches))
        cost = self.get_cost(new_possib_a, segment_a, possib_b, segment_b, enable_logging)
        if enable_logging:
            logging.log(
                logging.INFO,
                f"with intermediate note {intermediate_pitch}."
            )
        return cost

    @cache
    def get_cost(self, possib_a, segment_a, possib_b=None, segment_b=None, enable_logging=False):
        rules = self.transition_rules if possib_b else self.single_rules

        total_cost = 0
        for rule in rules:
            if possib_b is not None:
                cost = rule.get_int_cost(possib_a, possib_b, segment_a, segment_b)
            else:
                cost = rule.get_int_cost(possib_a, segment_a)
            if enable_logging and cost != 0:
                logging.log(logging.INFO, f"Cost += {cost} due to {rule.__class__.__name__}")
            total_cost += cost
            if total_cost == float('inf'):
                break

        if enable_logging and possib_b:
            logging.log(
                logging.INFO,
                f"Transition cost {possib_a} -> {possib_b}: {total_cost}."
            )
        elif enable_logging and not possib_b:
            logging.log(logging.INFO, f"Local cost {possib_a}: {total_cost}.")
        return total_cost

    def should_skip(self, segment: 'Segment') -> SkipDecision:
        return self.skip_rules.should_skip(segment)


class Rule(ABC):
    def __init__(self, cost):
        self.cost = cost

    def get_int_cost(self, *args, **kwargs):
        cost = self.get_cost(*args, **kwargs)
        if cost == float('inf') or cost == -float('inf'):
            return cost
        return int(cost)

    @abstractmethod
    def get_cost(self, *args, **kwargs):
        pass


class TransitionRule(Rule):
    @abstractmethod
    def get_cost(self, possib_a, possib_b, segment_a, segment_b):
        pass

    @cache
    def get_pairs(self, possib_a, possib_b):
        if len(possib_a) == len(possib_b):
            return [(a, b) for a, b in zip(possib_a, possib_b)]
        if len(possib_a) > len(possib_b):
            return [(b, a) for a, b in self.get_pairs(possib_b, possib_a)]
        assert len(possib_a) < len(possib_b)

        pairs = [(possib_a[0], possib_b[0]), (possib_a[-1], possib_b[-1])]

        # get all possible injective mappings
        middle_left = self._get_all_pair_possibs(possib_a[1:-1], possib_b[1:-1])
        possibs = []
        min_diff = float('inf')
        for possib in middle_left:
            diff = 0
            for a, b in possib:
                diff += self.distance_between(a, b)
            if diff < min_diff:
                possibs = possib
                min_diff = diff
        pairs += list(possibs)

        # match all unmatched nodes of B with A
        not_matched = set(possib_b)
        for a, b in pairs:
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
            pairs.append((closest, b))
        return pairs

    def _get_all_pair_possibs(self, possib_a, possib_b):
        if len(possib_a) == len(possib_b):
            return [list(zip(possib_a, possib_a))]

        assert len(possib_a) < len(possib_b)

        result = []
        n = len(possib_a)
        m = len(possib_b)
        for to_tuple in itertools.combinations(range(m), n):
            from_tuple = tuple(range(n))
            note_pairs = [(possib_a[f], possib_b[t]) for f, t in zip(from_tuple, to_tuple)]
            result.append(tuple(note_pairs))
        return result

    @staticmethod
    def distance_between(part_a, part_b):
        return math.ceil(max(abs(part_a - part_b) / 2 - 1, 0))


class ParallelFifths(TransitionRule):

    def get_cost(self, possib_a, possib_b, segment_a, segment_b):
        return self.cost if self.has_parallel_fifths(possib_a, possib_b) else 0

    @cache
    def has_parallel_fifths(self, possib_a: Possibility, possib_b: Possibility):
        pairs_list = self.get_pairs(possib_a.integer_pitches, possib_b.integer_pitches)

        for pair1Index in range(len(pairs_list)):
            (higherPitchA, higherPitchB) = pairs_list[pair1Index]
            for pair2Index in range(pair1Index + 1, len(pairs_list)):
                (lowerPitchA, lowerPitchB) = pairs_list[pair2Index]
                if abs(higherPitchA - lowerPitchA) % 12 == 7 and abs(higherPitchB - lowerPitchB) % 12 == 7:
                    return True

        return False


class ParallelOctaves(TransitionRule):
    def get_cost(self, possib_a, possib_b, segment_a, segment_b):
        return self.cost if self.has_parallel_octaves(possib_a.integer_pitches, possib_b.integer_pitches) else 0

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
        >>> C3 = Pitch('C3')
        >>> D3 = Pitch('D3')
        >>> G3 = Pitch('G3')
        >>> A3 = Pitch('A3')
        >>> C4 = Pitch('C4')
        >>> D4 = Pitch('D4')


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


        >>> B3 = Pitch('B3')
        >>> possibA2 = (C4, G3, C3)
        >>> possibB2 = (B3, A3, D3)
        >>> possibility.parallelOctaves(possibA2, possibB2)
        False
        '''
        pairsList = self.get_pairs(possibA, possibB)

        for pair1Index in range(len(pairsList)):
            (higherPitchA, higherPitchB) = pairsList[pair1Index]
            for pair2Index in range(pair1Index + 1, len(pairsList)):
                (lowerPitchA, lowerPitchB) = pairsList[pair2Index]
                if abs(higherPitchA - lowerPitchA) % 12 == 0 and abs(higherPitchB - lowerPitchB) % 12 == 0:
                    return True
        return False


def has_similar_motion(a1, a2, b1, b2):
    diff_h = a1 - a2
    diff_l = b1 - b2
    return ((diff_l > 0) == (diff_h > 0)) and diff_l != 0 and diff_h != 0


class HiddenFifth(TransitionRule):

    def get_cost(self, possib_a, possib_b, segment_a, segment_b):
        return self.cost if self.has_hidden_fifth(possib_a.integer_pitches, possib_b.integer_pitches) else 0

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
        >>> C3 = Pitch('C3')
        >>> D3 = Pitch('D3')
        >>> E3 = Pitch('E3')
        >>> F3 = Pitch('F3')
        >>> E5 = Pitch('E5')
        >>> A5 = Pitch('A5')


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


        >>> Ab5 = Pitch('A-5')
        >>> possibA2 = (E5, E3, C3)
        >>> possibB2 = (Ab5, F3, D3)
        >>> possibility.hiddenFifth(possibA2, possibB2)
        False


        Now, we have the soprano and bass parts again moving to A5 and D3, whose
        simple interval is a perfect fifth. However, the bass moves up while the
        soprano moves down. Therefore, there is no hidden fifth.


        >>> E6 = Pitch('E6')
        >>> possibA3 = (E6, E3, C3)
        >>> possibB3 = (A5, F3, D3)
        >>> possibility.hiddenFifth(possibA3, possibB3)
        False
        '''
        pairsList = self.get_pairs(possibA, possibB)
        (highestPitchA, highestPitchB) = pairsList[0]
        (lowestPitchA, lowestPitchB) = pairsList[-1]

        similar_motion = has_similar_motion(highestPitchA, highestPitchB, lowestPitchA, lowestPitchB)
        return abs(highestPitchB - lowestPitchB) % 12 == 7 and similar_motion


class HiddenOctave(TransitionRule):
    def get_cost(self, possib_a, possib_b, segment_a, segment_b):
        return self.cost if self.has_hidden_octave(possib_a.integer_pitches, possib_b.integer_pitches) else 0

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
        >>> C3 = Pitch('C3')
        >>> D3 = Pitch('D3')
        >>> E3 = Pitch('E3')
        >>> F3 = Pitch('F3')
        >>> A5 = Pitch('A5')
        >>> D6 = Pitch('D6')


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


        >>> A6 = Pitch('A6')
        >>> possibA2 = (A6, E3, C3)
        >>> possibB2 = (D6, F3, D3)
        >>> possibility.hiddenOctave(possibA2, possibB2)
        False
        '''
        pairsList = self.get_pairs(possibA, possibB)
        (highestPitchA, highestPitchB) = pairsList[0]
        (lowestPitchA, lowestPitchB) = pairsList[-1]

        similar_motion = has_similar_motion(highestPitchA, highestPitchB, lowestPitchA, lowestPitchB)
        return abs(highestPitchB - lowestPitchB) % 12 == 0 and similar_motion


class VoiceOverlap(TransitionRule):
    def get_cost(self, possib_a, possib_b, segment_a, segment_b):
        return self.cost if self.has_voice_overlap(possib_a.integer_pitches, possib_b.integer_pitches) else 0

    @cache
    def has_voice_overlap(self, possib_a, possib_b):
        all_pairs = self.get_pairs(possib_a, possib_b)
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
        >>> C4 = Pitch('C4')
        >>> D4 = Pitch('D4')
        >>> E4 = Pitch('E4')
        >>> F4 = Pitch('F4')
        >>> G4 = Pitch('G4')
        >>> C5 = Pitch('C5')


        Here, case #2 is demonstrated. There is overlap between the soprano and
        alto parts, because F4 in the soprano in possibB1 is lower than the G4
        in the alto in possibA1. Note that neither possibility has to have voice
        crossing for voice overlap to occur, as shown.


        >>> possibA1 = (C5, G4, E4, C4)
        >>> possibB1 = (F4, F4, D4, D4)
        >>> possibility.voiceOverlap(possibA1, possibB1)
        True

        Here is the same example as above, except the soprano of the second
        possibility is now B4, which does not overlap the G4 of the first.
        Now, there is no voice overlap.


        >>> B4 = Pitch('B4')
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


class MinimizeMovementsMiddleVoices(TransitionRule):
    def get_cost(self, possib_a, possib_b, segment_a, segment_b):
        return self.get_cached_cost(possib_a.integer_pitches, possib_b.integer_pitches)

    @cache
    def get_cached_cost(self, possib_a, possib_b):
        pairs = self.get_pairs(possib_a, possib_b)
        diff = sum([self.distance_between(a, b) for a, b in pairs])
        if diff == 0 and self.cost == float('inf'):
            return 0
        return self.cost * diff / max(len(possib_a), len(possib_b))


class MinimizeMovementsSopranoVoice(TransitionRule):
    def get_cost(self, possib_a, possib_b, segment_a, segment_b):
        diff = self.distance_between(possib_a.integer_pitches[0], possib_b.integer_pitches[0])
        if diff == 0 and self.cost == float('inf'):
            return 0
        else:
            return math.floor(diff / 2) * self.cost


class PartMovementsWithinLimits(TransitionRule):
    def __init__(self, cost, limits):
        super().__init__(cost)
        self.limits = limits

    def get_cost(self, possib_a, possib_b, segment_a, segment_b):
        return self.cost if not self.part_movements_within_limits(possib_a.pitches, possib_b.pitches) else 0

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
        >>> C4 = Pitch('C4')
        >>> D4 = Pitch('D4')
        >>> E4 = Pitch('E4')
        >>> F4 = Pitch('F4')
        >>> G4 = Pitch('G4')
        >>> A4 = Pitch('A4')
        >>> B4 = Pitch('B4')
        >>> C5 = Pitch('C5')

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


class UpperPartsSame(TransitionRule):
    def get_cost(self, possib_a, possib_b, segment_a, segment_b):
        return self.cost if not self.upper_parts_same(possib_a.pitches, possib_b.pitches) else 0

    def upper_parts_same(self, possibA, possibB):
        '''
        Returns True if the upper parts are the same.
        False otherwise.

        >>> from music21.figuredBass import possibility
        >>> C4 = Pitch('C4')
        >>> D4 = Pitch('D4')
        >>> E4 = Pitch('E4')
        >>> F4 = Pitch('F4')
        >>> G4 = Pitch('G4')
        >>> B4 = Pitch('B4')
        >>> C5 = Pitch('C5')
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


class PartsSame(TransitionRule):
    def __init__(self, cost, parts_to_check):
        super().__init__(cost)
        self.parts_to_check = parts_to_check

    def get_cost(self, possib_a, possib_b, segment_a, segment_b):
        return self.cost if not self.are_parts_same(possib_a, possib_b) else 0

    def are_parts_same(self, possib_a, possib_b):
        '''
        Takes in partsToCheck, a list of part numbers. Checks if pitches at those part numbers of
        possibA and possibB are equal, determined by pitch space.

        >>> from music21.figuredBass import possibility
        >>> C4 = Pitch('C4')
        >>> E4 = Pitch('E4')
        >>> G4 = Pitch('G4')
        >>> B4 = Pitch('B4')
        >>> C5 = Pitch('C5')
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


class UnpreparedNote(TransitionRule):
    def get_cost(self, possib_a, possib_b, segment_a, segment_b):
        return self.cost if self.has_unprepared_note(possib_a, possib_b, segment_b) else 0

    def has_unprepared_note(self, possib_a, possib_b, segment_b):
        seventh = None
        ninth = None
        segment_chord = possib_b.get_segment_option(segment_b).segment_chord
        pos = segment_chord.getChordStep(7)
        if pos:
            seventh = pos
        pos = segment_chord.getChordStep(7)
        if pos:
            ninth = pos
        if seventh is not None:
            for n2 in possib_b.get_pitches():
                if n2.pitchClass == seventh.pitchClass and int(n2.ps) not in possib_a.integer_pitches:
                    return True

        if ninth is not None:
            for n2 in possib_b.get_pitches():
                if n2.pitchClass == ninth.pitchClass and int(n2.ps) not in possib_a.integer_pitches:
                    return True
        return False


class CounterMovement(TransitionRule):
    def get_cost(self, possib_a, possib_b, segment_a, segment_b):
        possib_a, possib_b = possib_a.integer_pitches, possib_b.integer_pitches
        if not has_similar_motion(possib_a[0], possib_b[0], possib_a[-1], possib_b[-1]):
            return 0
        return self.cost


class SingleRule(Rule):
    @abstractmethod
    def get_cost(self, possib, segment):
        pass


class AvoidSeventhChord(SingleRule):
    def get_cost(self, possib, segment):
        segment_option = possib.get_segment_option(segment)
        if len(segment_option.segment_chord) <= 1:
            return 0
        classes = [p.pitchClass for p in segment_option.segment_chord[0].pitches]
        for pitch in possib.get_pitches():
            if pitch.pitchClass not in classes:
                return self.cost
        return 0


class IsPlayable(SingleRule):
    def get_cost(self, possib, segment):
        return self.cost if not self.is_playable(possib) else 0

    def is_playable(self, possib):
        if len(possib.integer_pitches) < 5:
            return self.playable_by_one_hand(possib.integer_pitches[:-1])
        else:
            return (self.playable_by_one_hand(possib.integer_pitches[:-2])
                    and self.playable_by_one_hand(possib.integer_pitches[-2:]))

    @staticmethod
    def playable_by_one_hand(notes):
        return notes[0] - notes[-1] <= 12


class NoSmallSecondInterval(SingleRule):
    def get_cost(self, possib, segment):
        for i in range(len(possib.integer_pitches) - 1):
            p1 = possib.integer_pitches[i]
            p2 = possib.integer_pitches[i + 1]
            if p1 - p2 <= 1:
                return self.cost
        return 0


class VoiceCrossing(SingleRule):
    def get_cost(self, possib, segment):
        return self.cost if self.voice_crossing(possib) else 0

    def voice_crossing(self, possib):
        for part1Index in range(len(possib.integer_pitches)):
            higherPitch = possib.integer_pitches[part1Index]
            for part2Index in range(part1Index + 1, len(possib.integer_pitches)):
                lowerPitch = possib.integer_pitches[part2Index]
                if higherPitch < lowerPitch:
                    return True

        return False


class HasDuplicate(SingleRule):
    def get_cost(self, possib, segment):
        return self.cost if len(possib.integer_pitches) != len(set(possib.integer_pitches)) else 0


class IsIncomplete(SingleRule):
    def get_cost(self, possib, segment):
        segment_option = possib.get_segment_option(segment)
        if not self.isIncomplete(
            possib.get_pitches(),
            segment_option.pitch_names_in_chord.copy(),
            segment.melody_pitches_at_strike
        ):
            return 0
        return self.cost

    def isIncomplete(self, possibA, pitchNamesToContain, melody_pitches):
        for pitch in melody_pitches:
            if pitch in pitchNamesToContain:
                pitchNamesToContain.remove(pitch)
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
    def get_cost(self, possib, segment):
        return self.cost if not self.upperPartsWithinLimit(possib.integer_pitches) else 0

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
        >>> C3 = Pitch('C3')
        >>> E3 = Pitch('E3')
        >>> E4 = Pitch('E4')
        >>> G4 = Pitch('G4')
        >>> C5 = Pitch('C5')
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
        return possibA[0] - possibA[-2] <= maxSemitoneSeparation


class PitchesUnderMelody(SingleRule):
    def get_cost(self, possib, segment):
        if len(segment.melody_pitches) == 0:
            return 0

        smallest_melody_pitch = min([pitch.ps for pitch in segment.melody_pitches])
        return max(0, self.cost * (possib.integer_pitches[0] - smallest_melody_pitch))


class PitchesWithinLimit(SingleRule):
    def get_cost(self, possib, segment):
        return self.cost if not self.pitchesWithinLimit(possib.integer_pitches) else 0

    def pitchesWithinLimit(self, possibA, maxPitch=Pitch('B5').ps, minRightHandPitch=Pitch('A3').ps):
        '''
        Returns True if all pitches in possibA are less than or equal to
        the maxPitch provided. Comparisons between pitches are done using pitch
        comparison methods, which are based on pitch space values
        (see :class:`~music21.Pitch`).


        Used in :class:`~music21.figuredBass.segment.Segment` to filter
        resolutions of special Segments which can have pitches exceeding
        the universal maxPitch of a :class:`~music21.figuredBass.realizer.FiguredBassLine`.


        >>> from music21.figuredBass import possibility
        >>> from music21.figuredBass import resolution
        >>> G2 = Pitch('G2')
        >>> D4 = Pitch('D4')
        >>> F5 = Pitch('F5')
        >>> B5 = Pitch('B5')
        >>> domPossib = (B5, F5, D4, G2)
        >>> possibility.pitchesWithinLimit(domPossib)
        True
        >>> resPossib = resolution.dominantSeventhToMajorTonic(domPossib)
        >>> resPossib  # Contains C6 > B5
        (<music21.Pitch C6>, <music21.Pitch E5>, <music21.Pitch C4>, <music21.Pitch C3>)
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


class NotTooLow(SingleRule):
    C3_INTEGER_PITCH = int(Pitch("C3").ps)

    def get_cost(self, possib, segment):
        bass_note = possib.integer_pitches[-1]
        if bass_note <= self.C3_INTEGER_PITCH and possib.integer_pitches[-2] - bass_note < 7:
            return self.cost
        return 0


class ContainRoot(SingleRule):
    def get_cost(self, possib, segment):
        segment_option = possib.get_segment_option(segment)
        root = segment_option.segment_chord.root()
        for p in possib.integer_pitches:
            if (p - root.ps) % 12 == 0:
                return 0
        return self.cost


class UseLeastAmountOfNotes(SingleRule):
    def __init__(self, cost, dynamic_ranges):
        super().__init__(cost)
        self.dynamic_ranges = dynamic_ranges

    def get_cost(self, possib, segment: 'Segment'):
        min_voices = self.dynamic_ranges[segment.dynamic][0]
        return max(0, (len(possib.integer_pitches) - min_voices)) * self.cost


class NotesFromFigures(SingleRule):
    def get_cost(self, possib: Possibility, segment):
        segment_option = possib.get_segment_option(segment)
        needed_pitch_classes = set(int(Pitch(p).ps) % 12 for p in segment_option.pitch_names_in_figures)
        for p in possib.integer_pitches:
            if p % 12 in needed_pitch_classes:
                needed_pitch_classes.remove(p % 12)
        return 0 if len(needed_pitch_classes) == 0 else self.cost


class DoubleRootIfCadence(SingleRule):
    def get_cost(self, possib, segment):
        if segment.ends_cadence and segment.duration.quarterLength >= 2:
            root_pitch = possib.integer_pitches[-1]
            for pitch in possib.integer_pitches[:-1]:
                if pitch - root_pitch % 12 == 0:
                    return 0
            return self.cost
        return 0


class OnlyAllowSixOptionAfterCadence(SingleRule):
    def get_cost(self, possib: Possibility, segment):
        if segment.prev_segment is None:
            return 0
        if segment.notation_strings[0] is None and possib.option_index != 0:
            if segment.prev_segment.ends_cadence:
                jump_down = segment.bassNote.pitch.ps - segment.prev_segment.bassNote.pitch.ps
                if jump_down != 1:
                    return self.cost
                else:
                    return 0
            else:
                return self.cost
        elif segment.notation_strings[0] is None and possib.option_index == 0:
            if segment.prev_segment.ends_cadence:
                jump_down = segment.bassNote.pitch.ps - segment.prev_segment.bassNote.pitch.ps
                if jump_down == 1:
                    return self.cost
                else:
                    return 0
            else:
                return 0

        return 0


class AvoidDoubling(SingleRule):
    def get_cost(self, possib: 'Possibility', segment: 'Segment'):
        segment_option = possib.get_segment_option(segment)
        num_pitches_possible = min(len(segment_option.pitch_names_in_chord), len(possib.integer_pitches))
        different_pitches = len(set(p % 12 for p in possib.integer_pitches))
        return max(num_pitches_possible - different_pitches, 0) * self.cost


def partPairs(possibA, possibB):
    '''
    Groups together pitches of possibA and possibB which correspond to the same part,
    constituting a shared part.

    >>> from music21.figuredBass import possibility
    >>> C4 = Pitch('C4')
    >>> D4 = Pitch('D4')
    >>> E4 = Pitch('E4')
    >>> F4 = Pitch('F4')
    >>> G4 = Pitch('G4')
    >>> B4 = Pitch('B4')
    >>> C5 = Pitch('C5')
    >>> possibA1 = (C5, G4, E4, C4)
    >>> possibB1 = (B4, F4, D4, D4)
    >>> possibility.partPairs(possibA1, possibA1)
    [(<music21.Pitch C5>, <music21.Pitch C5>), (<music21.Pitch G4>, <music21.Pitch G4>), (<music21.Pitch E4>, <music21.Pitch E4>), (<music21.Pitch C4>, <music21.Pitch C4>)]
    >>> possibility.partPairs(possibA1, possibB1)
    [(<music21.Pitch C5>, <music21.Pitch B4>), (<music21.Pitch G4>, <music21.Pitch F4>), (<music21.Pitch E4>, <music21.Pitch D4>), (<music21.Pitch C4>, <music21.Pitch D4>)]

    '''
    return list(zip(possibA, possibB))
