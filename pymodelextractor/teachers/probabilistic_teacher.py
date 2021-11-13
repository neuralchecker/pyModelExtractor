from abc import ABC, abstractmethod

from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.sequence import Sequence
from pythautomata.automata.wheighted_automaton_definition.weighted_automaton import WeightedAutomaton
from pymodelextractor.utilities import pdfa_utils
from typing import Union

class ProbabilisticTeacher(ABC):

    def __init__(self, tolerance):
        self.tolerance = tolerance
        self._equivalence_queries_count = 0
        self._last_token_weight_queries_count = 0

    @abstractmethod
    def sequence_weight(self, sequence: Sequence) -> float:
        raise NotImplementedError

    @abstractmethod
    def log_sequence_weight(self, sequence: Sequence) -> float:
        raise NotImplementedError

    @abstractmethod
    def last_token_weights(self, sequence: Sequence, required_suffixes: list[Sequence]) -> list[float]:
        raise NotImplementedError

    def equivalence_query(self, aut: WeightedAutomaton) -> tuple[bool, Union[Sequence,None]]:
        tried = set()
        suffixes = list()
        suffixes.append(self.terminal_symbol)
        for symbol in self.alphabet.symbols:
            suffixes.append(Sequence([symbol]))
        rand_words = pdfa_utils.get_test_data(self.alphabet, 5000)
        for word in rand_words:
            prefixes = sorted(word.get_prefixes(), key=len)
            for prefix in prefixes:
                if prefix not in tried:
                    obs1 = self.last_token_weights(prefix, suffixes)
                    obs2 = aut.get_last_token_weights(prefix, suffixes)
                    if not pdfa_utils.are_within_tolerance_limit(obs1, obs2, self.tolerance):
                        return False, prefix
                    tried.add(prefix)
        return True, None

    def reset(self) -> None:
        self._equivalence_queries_count = 0
        self._last_token_weight_queries_count = 0

    def log_probability_error(self, seq, aut: WeightedAutomaton) -> float:
        return abs(aut.log_sequence_weight(seq) - self.log_sequence_weight(seq))

    @property
    @abstractmethod
    def alphabet(self) -> Alphabet:
        raise NotImplementedError

    @property
    @abstractmethod
    def terminal_symbol(self) -> Sequence:
        raise NotImplementedError

    @property
    def equivalence_queries_count(self):
        return self.equivalence_queries_count

    @property
    def last_token_weight_queries_count(self):
        return self.equivalence_queries_count
