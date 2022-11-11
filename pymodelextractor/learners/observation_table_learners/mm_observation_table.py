from typing import Union
from symtable import Symbol
from typing import Sequence
from pythautomata.base_types.alphabet import Alphabet

from pymodelextractor.learners.observation_table_learners.observation_table import TableInconsistency


class MMObservationTable:
    red: set[Sequence]
    blue: set[Sequence]
    observations: dict[Sequence, list[Symbol]]
    exp: list[Sequence]

    def __init__(self):
        self.red = set()
        self.blue = set()
        self.observations = {}
        self.exp = []

        self.redValues = set()

    def __getitem__(self, sequence: Sequence) -> list[Symbol]:
        return self.observations[sequence]

    def __setitem__(self, sequence: Sequence, observationsRow: list[Symbol]):
        self.observations[sequence] = observationsRow

    def is_closed(self) -> Union[list[Symbol], None]:
        for sequence in self.blue:
            blue_symbol = self.observations[sequence]
            if self.redValues.get(blue_symbol) == None:
                return blue_symbol
        return None

    def _get_red_values(self) -> set[list[Symbol]]:
        self.redValues = set()
        for sequence in self.red: 
            redSymbol = self.observations[sequence]
            if not (self.redValues in redSymbol):
                self.redValues.add(redSymbol)
        return self.redValues

    def find_inconsistency(self, alphabet: Alphabet) -> Union[TableInconsistency, None]:
        redValues: dict[list[Symbol], Sequence]
        redValues = {}
        for row in self.red:
            rowSymbols = self.observations[row]
            redValue = redValues.get(rowSymbols)
            if redValue != None:
                inconsistency = self._are_inconsistent(redValue, row, alphabet)
                if inconsistency != None:
                    return inconsistency
            else:
                redValues[rowSymbols] = row
        return None

    def _are_inconsistent(self, sequence1, sequence2, alphabet: Alphabet)-> \
            Union[TableInconsistency, None]:
        for symbol in alphabet:
            suffixedSequence1 = sequence1 + symbol
            suffixedSequence2 = sequence2 + symbol
            if self.observations[suffixedSequence1] != self.observations[suffixedSequence2]:
                return TableInconsistency(sequence1, sequence2, symbol, None)
        return None
    
    def move_from_blue_to_red(self, sequence: Sequence):
        self.blue.remove(sequence)
        self.add_to_red(sequence, self.observations[sequence])

    def add_to_red(self, sequence: Sequence, values: list[Symbol]):
        self.red.add(sequence)
        self.redValues.add(values)
    
    def add_to_blue(self, sequence: Sequence):
        self.blue.add(sequence)

    def __str__(self):
        lines = ["\nObservation Table:",
                 "\n=================",
                 "\nRED: " + repr(self.red),
                 "\nBLUE: " + repr(self.blue),
                 "\nEXP: " + repr(self.exp),
                 "\nOBSERVATIONS: " + repr(self.observations) + "\n"]
        return ''.join(lines)