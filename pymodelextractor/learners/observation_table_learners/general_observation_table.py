from typing import Union
from symtable import Symbol
from typing import Sequence
from pythautomata.base_types.alphabet import Alphabet

from pymodelextractor.learners.observation_table_learners.observation_table import TableInconsistency
from pymodelextractor.teachers.general_teacher import GeneralTeacher


class GeneralObservationTable:
    red: set[Sequence]
    blue: set[Sequence]
    observations: dict[Sequence, Union[list[Symbol], list[bool]]]
    exp: list[Sequence]

    def __init__(self):
        self.red = set()
        self.blue = set()
        self.observations = {}
        self.exp = []

        self.redValues = set()

    def __getitem__(self, sequence: Sequence) -> Union[list[Symbol], list[bool]]:
        return self.observations[sequence]

    def __setitem__(self, sequence: Sequence, observationsRow: Union[list[Symbol], list[bool]]):
        self.observations[sequence] = observationsRow

    def is_closed(self) -> Union[Sequence, None]:
        for sequence in self.blue:
            blue_symbol = self.observations[sequence]
            if not(tuple(blue_symbol) in self.redValues):
                return sequence
        return None

    def update_red_values(self) -> set[Union[list[Symbol], list[bool]]]:
        self.redValues = set()
        for sequence in self.red: 
            redSymbol = tuple(self.observations[sequence])
            if not (self.redValues in redSymbol):
                self.redValues.add(redSymbol)
        return self.redValues

    def find_inconsistency(self, alphabet: Alphabet) -> Union[TableInconsistency, None]:
        redValues: dict[tuple, Union[Sequence, bool]]
        redValues = {}
        for row in self.red:
            rowSymbols = self.observations[row]
            redValue = redValues.get(tuple(rowSymbols))
            if not (redValue is None):
                inconsistency = self._are_inconsistent(redValue, row, alphabet)
                if not (inconsistency is None):
                    return inconsistency
            else:
                redValues[tuple(rowSymbols)] = row
        return None

    def _are_inconsistent(self, sequence1, sequence2, alphabet: Alphabet)-> \
            Union[TableInconsistency, None]:
        for symbol in alphabet.symbols:
            suffixedSequence1 = sequence1 + symbol
            suffixedSequence2 = sequence2 + symbol
            if self.observations[suffixedSequence1] != self.observations[suffixedSequence2]:
                differenceSequence = self._observation_difference_between(
                    suffixedSequence1, suffixedSequence2)
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

    def move_from_blue_to_red(self, sequence: Sequence):
        self.blue.remove(sequence)
        self.add_to_red(sequence, self.observations[sequence])

    def add_to_red(self, sequence: Sequence, values: Union[list[Symbol], list[bool]]):
        self.red.add(sequence)
        self.redValues.add(tuple(values))
    
    def add_to_blue(self, sequence: Sequence):
        self.blue.add(sequence)

    def fill_observations(self, oracle: GeneralTeacher):
        for red_seq in self.red:
            if red_seq not in self.observations or \
                len(self.observations[red_seq]) != len(self.exp):
                sequences = []
                for suffix in self.exp:
                    sequence = red_seq + suffix
                    sequences.append(oracle.membership_query(sequence))

                self.observations[red_seq] = sequences

        for blue_seq in self.blue:
            if blue_seq not in self.observations or \
                len(self.observations[blue_seq]) != len(self.exp):
                sequences = []
                for suffix in self.exp:
                    sequence = blue_seq + suffix
                    sequences.append(oracle.membership_query(sequence))

                self.observations[blue_seq] = sequences

    def __str__(self):
        lines = ["\nObservation Table:",
                 "\n=================",
                 "\nRED: " + repr(self.red),
                 "\nBLUE: " + repr(self.blue),
                 "\nEXP: " + repr(self.exp),
                 "\nOBSERVATIONS: " + repr(self.observations) + "\n"]
        return ''.join(lines)