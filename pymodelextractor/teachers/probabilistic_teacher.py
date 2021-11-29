from abc import ABC, abstractmethod

from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.sequence import Sequence
from pythautomata.automata.wheighted_automaton_definition.weighted_automaton import WeightedAutomaton

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
        raise NotImplementedError

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
        return self._equivalence_queries_count

    @property
    def last_token_weight_queries_count(self):
        return self._last_token_weight_queries_count
