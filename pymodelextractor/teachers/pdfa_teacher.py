from pythautomata.base_types.sequence import Sequence
from pythautomata.automata.wheighted_automaton_definition.weighted_automaton import WeightedAutomaton

from pymodelextractor.teachers.probabilistic_teacher import ProbabilisticTeacher


class PDFATeacher(ProbabilisticTeacher):

    def __init__(self, model: WeightedAutomaton, tolerance):
        super().__init__(tolerance)
        self.__model = model

    def sequence_weight(self, sequence: Sequence):
        return self.__model.sequence_weight(sequence)

    def log_sequence_weight(self, sequence: Sequence):
        return self.__model.log_sequence_weight(sequence)

    def last_token_weights(self, sequence: Sequence, required_suffixes: list[Sequence]):
        self._last_token_weight_queries_count +=1        
        return self.__model.get_last_token_weights(sequence, required_suffixes)

    def get_log_probability_error(self, seq, aut: WeightedAutomaton):
        return abs(aut.log_sequence_weight(seq) - self.log_sequence_weight(seq))

    @property
    def alphabet(self):
        return self.__model.alphabet

    @property
    def terminal_symbol(self):
        return self.__model.terminal_symbol
