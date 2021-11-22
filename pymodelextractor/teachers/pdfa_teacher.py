from pythautomata.base_types.sequence import Sequence
from pythautomata.automata.wheighted_automaton_definition.weighted_automaton import WeightedAutomaton

from pymodelextractor.teachers.probabilistic_teacher import ProbabilisticTeacher
from typing import Union

class PDFATeacher(ProbabilisticTeacher):

    def __init__(self, model: WeightedAutomaton, tolerance: float, comparison_strategy: PDFAComparator):
        super().__init__(tolerance)
        self._comparison_strategy = comparison_strategy
        self.__target_pdfa_model = model

    def sequence_weight(self, sequence: Sequence):
        return self.__target_pdfa_model.sequence_weight(sequence)

    def log_sequence_weight(self, sequence: Sequence):
        return self.__target_pdfa_model.log_sequence_weight(sequence)

    def last_token_weights(self, sequence: Sequence, required_suffixes: list[Sequence]):
        self._last_token_weight_queries_count +=1        
        return self.__target_pdfa_model.get_last_token_weights(sequence, required_suffixes)

    def get_log_probability_error(self, seq, aut: WeightedAutomaton):
        return abs(aut.log_sequence_weight(seq) - self.log_sequence_weight(seq))
    
    def equivalence_query(self, aut: WeightedAutomaton) -> tuple[bool, Union[Sequence,None]]:
        self._equivalence_queries_count += 1
        counterexample = self._comparison_strategy.get_counterexample_between(aut, self.__target_pdfa_model)
        are_equivalent = counterexample is None
        return (are_equivalent, counterexample)

    @property
    def alphabet(self):
        return self.__model.alphabet

    @property
    def terminal_symbol(self):
        return self.__model.terminal_symbol
