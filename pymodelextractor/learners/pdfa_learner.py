from abc import ABC, abstractmethod
from pythautomata.automata.wheighted_automaton_definition.weighted_automaton import WeightedAutomaton

from pymodelextractor.teachers.probabilistic_teacher import ProbabilisticTeacher


class PDFALearner(ABC):

    @abstractmethod
    def learn(self, teacher: ProbabilisticTeacher) -> WeightedAutomaton:
        """
        Function that learns a Probabilistic Deterministic Finite Automaton performing queries to a teacher
        Parameters
        ----------
        teacher : ProbabilisticTeacher

        Returns
        -------
        WeightedAutomaton
        """
        raise NotImplementedError
