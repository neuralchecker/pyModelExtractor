from ctypes import Union
from symtable import Symbol
from typing import Sequence
from pythautomata.base_types.alphabet import Alphabet

from pymodelextractor.learners.observation_table_learners.observation_table import TableInconsistency


class MMObservationTable:
    red: set[Sequence]
    blue: set[Sequence]
    observations: dict[Sequence, list[Symbol]]

    def __getitem__(self, sequence: Sequence) -> list[bool]:
        return self.observations[sequence]

    def __setitem__(self, sequence: Sequence, observationsRow: list[bool]):
        self.observations[sequence] = observationsRow

    def is_closed(self) -> bool:
        raise NotImplementedError

    def find_inconsistency(self, alphabet: Alphabet) -> Union[TableInconsistency, None]:
        raise NotImplementedError

    def move_from_blue_to_red(self, sequence: Sequence):
        self.blue.remove(sequence)
        self.red.add(sequence)

    def _inconsistency_between(self, sequence1: Sequence, sequence2: Sequence, alphabet: Alphabet) -> \
            Union[TableInconsistency, None]:
        for symbol in alphabet.symbols:
            suffixedSequence1 = sequence1 + symbol
            suffixedSequence2 = sequence2 + symbol
            differenceSequence = self._observation_difference_between(
                suffixedSequence1, suffixedSequence2)
            if differenceSequence is not None:
                return TableInconsistency(sequence1, sequence2, symbol, differenceSequence)
        return None

    def _observation_difference_between(self, sequence1: Sequence, sequence2: Sequence) -> Union[Sequence, None]:
        observations1 = self.observations[sequence1]
        observations2 = self.observations[sequence2]
        assert len(observations1) == len(observations2)
        for i in range(0, len(observations1)):
            if observations1[i] != observations2[i]:
                return self.exp[i]
        return None

    def __str__(self):
        lines = ["\nObservation Table:",
                 "\n=================",
                 "\nRED: " + repr(self.red),
                 "\nBLUE: " + repr(self.blue),
                 "\nEXP: " + repr(self.exp),
                 "\nOBSERVATIONS: " + repr(self.observations) + "\n"]
        return ''.join(lines)