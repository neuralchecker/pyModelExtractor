from abc import ABC, abstractmethod

from pythautomata.automata.wheighted_automaton_definition.weighted_automaton import WeightedAutomaton
from pythautomata.base_types.symbol import Symbol

from pymodelextractor.learners.observation_table_learners.pdfa_observation_table \
    import PDFAObservationTable


class PDFAObservationTableTranslator(ABC):
    @abstractmethod
    def translate(self, observation_table: PDFAObservationTable, tolerance: float, terminal_symbol: Symbol) \
            -> WeightedAutomaton:
        """
        Function to translate from a PDFAObservation table into a Probabilistic Deterministic Finite Automaton
        Parameters
        ----------
        observation_table : PDFAObservation
        tolerance : float
        terminal_symbol : Sequence

        Returns
        -------
        WeightedAutomaton
        """
        raise NotImplementedError
