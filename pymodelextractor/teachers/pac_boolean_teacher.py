from typing import Tuple
from pymodelextractor.teachers.teacher import Teacher
from pythautomata.utilities.sequence_generator import SequenceGenerator
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.sequence import Sequence
from pythautomata.abstract.boolean_model import BooleanModel
from math import ceil, log, factorial, comb
import numpy as np

class PACBooleanTeacher(Teacher):

    def __init__(self, model: BooleanModel, epsilon: float, delta: float, sequence_generator: SequenceGenerator = None, max_seq_length: float = 128,
                 compute_epsilon_star: bool = True, verbose: bool = False):
        self.__target_model = model
        self._epsilon = epsilon
        self._delta = delta
        self.sample_size = 0
        self.epsilon_star = 0
        self.__compute_epsilon_star = compute_epsilon_star
        if sequence_generator is None:
            self._sequence_generator = SequenceGenerator(self.__target_model.alphabet, max_seq_length= max_seq_length)
        else:
            self._sequence_generator = sequence_generator
        self._verbose = verbose
        super().__init__()

    @property
    def alphabet(self) -> Alphabet:
        return self.__target_model.alphabet

    @property
    def membership_queries_count(self):
        return self._membership_queries_count

    @property
    def equivalence_queries_count(self):
        return self._equivalence_queries_count

    def membership_query(self, sequence: Sequence) -> bool:
        self._membership_queries_count +=1
        return self.__target_model.accepts(sequence)

    def equivalence_query(self, model: BooleanModel) -> Tuple[bool, Sequence]:
        self._equivalence_queries_count += 1
        if self._verbose: print("*** Equivalence Query - teacher counter:", self.equivalence_queries_count, "***")

        sample_size = self._calculate_sample_size()
        if self._verbose: print("Sample Size:", sample_size, "Epsilon:", self._epsilon, "Epsilon*:", self.epsilon_star, "Delta:", self._delta)

        errorCount = 0
        counterexample = None
        
        sequences = self._sequence_generator.generate_words(sample_size)
        np.sort(sequences)
        for sequence in sequences:
            if self.__target_model.accepts(sequence) != model.accepts(sequence):
                if not self.__compute_epsilon_star:
                    return (False, sequence)
                errorCount += 1
                if counterexample is None or len(sequence) < len(counterexample):
                    counterexample = sequence
        if errorCount > 0:
            self._calculate_epsilon_star_with(errorCount)        
        return (counterexample is None, counterexample)

    def _calculate_sample_size(self):        
        numberOfCalls = self.equivalence_queries_count
        sample_size = ceil((log(2) * (numberOfCalls + 1) - log(self._delta)) / self._epsilon)
        self.last_sample_size = sample_size
        return sample_size

    def _calculate_epsilon_star_with(self, errorCount: int):
        #combinations = factorial(self.last_sample_size) // factorial(errorCount) // factorial(self.last_sample_size - errorCount)
        combinations = comb(self.last_sample_size, errorCount)
        if (self.last_sample_size - errorCount) == 0:
            self.epsilon_star = float('inf')
        else:
            self.epsilon_star = (log(combinations) - log(self._delta)) / (self.last_sample_size - errorCount)
   

    def reset_statistics(self):
            self.sample_size = 0
            self.epsilon_star = 0
            self.last_sample_size = 0
            self.membership_queries_count = 0
            self.equivalence_queries_count = 0
            self._sequence_generator.reset_seed()
