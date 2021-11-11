from abc import ABC, abstractmethod

from pythautomata.abstract.finite_automaton import FiniteAutomaton
from pythautomata.base_types.alphabet import Alphabet
from pymodelextractor.learners.observation_table_learners.observation_table \
    import ObservationTable


class ObservationTableTranslator(ABC):
    @abstractmethod
    def translate(self, observation_table: ObservationTable, alphabet: Alphabet) -> FiniteAutomaton:
        raise NotImplementedError
