from abc import ABC, abstractmethod
from typing import Tuple
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.sequence import Sequence
from pythautomata.abstract.finite_automaton import FiniteAutomaton


class Teacher(ABC):

    def __init__(self):        
        self._equivalence_queries_count: int = 0
        self._membership_queries_count: int = 0

    @property
    @abstractmethod
    def alphabet(self) -> Alphabet:
        raise NotImplementedError

    @property
    @abstractmethod
    def membership_queries_count(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def equivalence_queries_count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def membership_query(self, sequence: Sequence) -> bool:
        raise NotImplementedError

    @abstractmethod
    def equivalence_query(self, automaton: FiniteAutomaton) -> Tuple[bool, Sequence]:
        """Checks whether the models are equivalent or not

        Args:
            automaton (FiniteAutomaton): target automaton to check whether it is equivalent to hidden model or not

        Returns:
            Tuple[bool, Sequence]: either (True, None) or (False, counter_example)
        """
        pass

    @abstractmethod
    def reset_statistics(self) -> None:
        raise NotImplementedError
