from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.alphabet import Alphabet
from collections import namedtuple
from typing import Union
import heapq
from pymodelextractor.utilities import pdfa_utils

epsilon = Sequence()

Inconsistency = namedtuple('Inconsistency', 'sequence1 sequence2 symbol different_suffix')


class PDFAObservationTable:

    def __init__(self, alphabet: Alphabet, tolerance: float):
        self.alphabet = alphabet
        self.red = set()
        self.__blue = set()
        self.__blue_queue = []
        self.__suffixes = []
        self.__suffixes_set = set()
        self.__observations = {}
        self.__tolerance = tolerance
        self.symbols = alphabet.symbols

    def __getitem__(self, element):
        return self.__observations[element]

    def __setitem__(self, sequence: Sequence, observations_row):
        self.__observations[sequence] = observations_row

    def add_suffix(self, sequence: Sequence) -> bool:
        added = False
        if sequence not in self.__suffixes_set:
            self.__suffixes_set.add(sequence)
            self.__suffixes.append(sequence)
            added = True
        return added


    def add_to_red(self, sequence: Sequence) -> None:
        self.red.add(sequence)

    def add_to_blue(self, weighted_sequence: tuple[float, Sequence]) -> None:
        heapq.heappush(self.__blue_queue, weighted_sequence)
        self.__blue.add(weighted_sequence[1])

    def remove_from_blue(self, sequence: Sequence) -> None:
        self.__blue.remove(sequence)
        self.__blue_queue = list(filter(lambda x: x[1] != sequence, self.__blue_queue))
        heapq.heapify(self.__blue_queue)

    def contains_in_red(self, sequence: Sequence) -> bool:
        return sequence in self.red

    def contains_in_blue(self, sequence: Sequence) -> bool:
        return sequence in self.__blue

    def get_suffixes(self) -> list[Sequence]:
        return self.__suffixes

    def contains_observation(self, sequence: Sequence) -> bool:
        return sequence in self.__observations

    def get_observed_sequences(self) -> list[Sequence]:
        return list(self.__observations.keys())

    def get_violating_closedness_sequence(self) -> Union[Sequence, None]:
        violating_sequence = None
        non_violating_seqs = []
        while len(self.__blue_queue) > 0 and violating_sequence is None:
            value, blue_sequence = heapq.heappop(self.__blue_queue)
            if not self.__same_observation_exists_in_red(blue_sequence):
                self.__blue.remove(blue_sequence)
                violating_sequence = blue_sequence
            else:
                non_violating_seqs.append((value, blue_sequence))
        for non_violating_seq in non_violating_seqs:
            heapq.heappush(self.__blue_queue, non_violating_seq)
        return violating_sequence

    def __same_observation_exists_in_red(self, blue_sequence):
        return any(pdfa_utils.are_within_tolerance_limit(self[blue_sequence], self[sequence], self.__tolerance)
                   for sequence in self.red)

    def find_inconsistency(self) -> Union[Inconsistency, None]:
        red_list = sorted(list(self.red))
        red_list_length = len(red_list)
        for i in range(red_list_length):
            for j in range(i + 1, red_list_length):
                red1 = red_list[i]
                red2 = red_list[j]
                if red1 != red2 and \
                        pdfa_utils.are_within_tolerance_limit(self.__observations[red1], self.__observations[red2],
                                                              self.__tolerance):
                    inconsistency = self.__inconsistency_between(red1, red2)
                    if inconsistency is not None:
                        return inconsistency
        return None

    def __inconsistency_between(self, sequence1: Sequence, sequence2: Sequence):
        for symbol in self.symbols:
            suffixed_sequence1 = sequence1 + symbol
            suffixed_sequence2 = sequence2 + symbol
            different_suffix = self.__max_difference(suffixed_sequence1, suffixed_sequence2)
            if different_suffix is not None:
                return Inconsistency(sequence1, sequence2, symbol, different_suffix)
        return None

    def __max_difference(self, sequence1: Sequence, sequence2: Sequence):
        observations1 = self.__observations[sequence1]
        observations2 = self.__observations[sequence2]
        assert len(observations1) == len(observations2)
        maxim = self.__tolerance
        max_i = 0
        for i in range(0, len(observations1)):
            diff = abs(observations1[i] - observations2[i])
            if diff > maxim:
                maxim = diff
                max_i = i
        if maxim == self.__tolerance:
            return None
        else:
            return self.__suffixes[max_i]

    def get_red_observations(self) -> dict[Sequence, list[float]]:
        return {key: self.__observations[key] for key in list(self.red)}
