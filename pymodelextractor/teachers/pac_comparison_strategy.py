from typing import Union
from pythautomata.utilities.uniform_length_sequence_generator import UniformLengthSequenceGenerator
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.sequence import Sequence
from pythautomata.abstract.model import Model
from pythautomata.abstract.boolean_model import BooleanModel
from pythautomata.utilities.sequence_generator import SequenceGenerator
from math import ceil, log, comb
import numpy as np


class PACComparisonStrategy:
    def __init__(self, target_model_alphabet: Alphabet, epsilon: float, 
                delta: float, max_seq_length: int = 128, 
                compute_epsilon_star: bool = True, sequence_generator: SequenceGenerator = None):
        self._epsilon = epsilon
        self._delta = delta
        self._equivalence_queries_count = 0
        self._compute_epsilon_star = compute_epsilon_star

        if sequence_generator is None:
            self._sequence_generator = UniformLengthSequenceGenerator(target_model_alphabet, 
                                                                      max_seq_length)
        else:
            self._sequence_generator = sequence_generator

    def get_counterexample_between(self, model: Union[Model, BooleanModel], 
                                   target_model: Union[Model, BooleanModel]) -> Sequence:
            self._equivalence_queries_count += 1
            sample_size = self._calculate_sample_size()
            sequences = self._sequence_generator.generate_words(sample_size)
            error_count = 0
            counterexample = None

            np.sort(sequences)
            for sequence in sequences:
                if target_model.process_query(sequence) != model.process_query(sequence):
                    if not self._compute_epsilon_star:
                        return (False, sequence)
                    error_count += 1
                    if counterexample is None or len(sequence) < len(counterexample):
                        counterexample = sequence
            if error_count > 0:
                self._calculate_epsilon_star_with(error_count)
            return counterexample
    
    def _calculate_sample_size(self):
        numberOfCalls = self._equivalence_queries_count
        sample_size = ceil((log(2) * (numberOfCalls + 1) - log(self._delta)) / self._epsilon)
        self.last_sample_size = sample_size
        return sample_size

    def _calculate_epsilon_star_with(self, errorCount: int):
        combinations = comb(self.last_sample_size, errorCount)
        if (self.last_sample_size - errorCount) == 0:
            self.epsilon_star = float('inf')
        else:
            self.epsilon_star = (log(combinations) - log(self._delta)) \
                 / (self.last_sample_size - errorCount)
