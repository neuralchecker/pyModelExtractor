from pythautomata.base_types.sequence import Sequence
from pythautomata.automata.wheighted_automaton_definition.weighted_automaton import WeightedAutomaton
from pythautomata.abstract.probabilistic_model import ProbabilisticModel
from pythautomata.utilities.sequence_generator import SequenceGenerator
from pymodelextractor.teachers.probabilistic_teacher import ProbabilisticTeacher
from pymodelextractor.utilities import pdfa_utils
from math import ceil, log, factorial
from typing import Union
import numpy as np

class PACProbabilisticTeacher(ProbabilisticTeacher):

    def __init__(self, model: ProbabilisticModel, epsilon: float, delta:float,
     tolerance: float, sequence_generator: SequenceGenerator, comparison_strategy: PDFAComparator, 
     compute_epsilon_star = False):

        super().__init__(tolerance)
        self._comparison_strategy = comparison_strategy
        self.__target_model = model
        self._epsilon = epsilon
        self._delta = delta
        self.sample_size = 0
        self.epsilon_star = 0
        self.__compute_epsilon_star = compute_epsilon_star
        self._sequence_generator = sequence_generator


    def sequence_weight(self, sequence: Sequence):
        return self.__target_model.sequence_weight(sequence)

    def log_sequence_weight(self, sequence: Sequence):
        return self.__target_model.log_sequence_weight(sequence)

    def last_token_weights(self, sequence: Sequence, required_suffixes: list[Sequence]):
        self._last_token_weight_queries_count +=1        
        return self.__target_model.get_last_token_weights(sequence, required_suffixes)

    def get_log_probability_error(self, seq, aut: WeightedAutomaton):
        return abs(aut.log_sequence_weight(seq) - self.log_sequence_weight(seq))
    
    def equivalence_query(self, aut: WeightedAutomaton) -> tuple[bool, Union[Sequence,None]]:        
        self._equivalence_queries_count += 1
        sample_size = self._calculate_sample_size()
        errorCount = 0
        counterexample = None      
        suffixes = []
        
        suffixes.append(self.terminal_symbol)
        total_error = 0
        for symbol in self.alphabet.symbols:
            suffixes.append(Sequence(symbol))
        
        rand_words = self._sequence_generator.generate_words(self.sample_size)
        np.sort(rand_words)
        counterexample = None
        for word in rand_words:            
            #print(prefix)
            obs1 = self.get_last_token_weights(word, suffixes)
            obs2 = aut.get_last_token_weights(word, suffixes)
            error = self.get_log_probability_error(word, aut)
            total_error += error
            if not pdfa_utils.are_within_tolerance_limit(obs1, obs2, self.tolerance):
                errorCount += 1
                if counterexample is None:
                    counterexample = word 
                if not self.__calculate_epsilon_star:
                    return False, counterexample               
        if errorCount > 0:
            self._calculate_epsilon_star_with(errorCount)

        return counterexample is None, counterexample

    @property
    def alphabet(self):
        return self.__model.alphabet

    @property
    def terminal_symbol(self):
        return self.__model.terminal_symbol

    def _calculate_sample_size(self):
        numberOfCalls = self.equivalence_queries_count
        self.sample_size = ceil((log(2) * (numberOfCalls + 1) - log(self._delta)) / self._epsilon)
        self.epsilon_star = -log(self._delta) / self.sample_size

    def _calculate_epsilon_star_with(self, errorCount: int):
        combinations = factorial(self.sample_size) // factorial(errorCount) // factorial(self.sample_size - errorCount)
        self.epsilon_star = (log(combinations) - log(self._delta)) / (self.sample_size - errorCount)

