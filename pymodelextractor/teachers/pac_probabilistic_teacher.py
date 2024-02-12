from pythautomata.base_types.sequence import Sequence
from pythautomata.automata.wheighted_automaton_definition.weighted_automaton import WeightedAutomaton
from pythautomata.abstract.probabilistic_model import ProbabilisticModel
from pythautomata.utilities.sequence_generator import SequenceGenerator
from pythautomata.utilities.uniform_length_sequence_generator import UniformLengthSequenceGenerator
from pymodelextractor.teachers.probabilistic_teacher import ProbabilisticTeacher
from pythautomata.abstract.finite_automaton import FiniteAutomataComparator
from math import ceil, log, comb
from typing import Union
from pymodelextractor.utils.data_loader import DataLoader


class PACProbabilisticTeacher(ProbabilisticTeacher):

    def __init__(self, model: ProbabilisticModel, comparator: FiniteAutomataComparator, epsilon: float = 0.05,
                 delta: float = 0.01, sequence_generator: SequenceGenerator = None, max_seq_length: int = 128,
                 compute_epsilon_star: bool = True, parallel_cache = False, max_query_elements = 1_000_000, batch_size = 10_000, cache_from_dataloader:DataLoader = None):
        super().__init__(model, parallel_cache, max_query_elements, batch_size, cache_from_dataloader)
        self._comparator = comparator
        self._epsilon = epsilon
        self._delta = delta
        self.sample_size = 0
        self.epsilon_star = 0
        self._compute_epsilon_star = compute_epsilon_star        
        if sequence_generator is None:
            self._sequence_generator = UniformLengthSequenceGenerator(self._target_model.alphabet, max_seq_length= max_seq_length)
        else:
            self._sequence_generator = sequence_generator




    def log_sequence_weight(self, sequence: Sequence):
        return self._target_model.log_sequence_weight(sequence)

    def get_log_probability_error(self, seq, aut: WeightedAutomaton):
        return abs(aut.log_sequence_weight(seq) - self.log_sequence_weight(seq))
    
    def equivalence_query(self, aut: WeightedAutomaton) -> tuple[bool, Union[Sequence,None]]:        
        self._equivalence_queries_count += 1
        sample_size = self._calculate_sample_size()
        errorCount = 0
        counterexample = None      
        suffixes = []
        
        suffixes.append(self.terminal_symbol)
        #total_error = 0
        for symbol in self.alphabet.symbols:
            suffixes.append(Sequence((symbol,)))
        
        rand_words = self._sequence_generator.generate_words(sample_size)
        rand_words.sort(key=len)
        counterexample = None
        for word in rand_words:  
            obs1 = self._target_model.get_last_token_weights(word, suffixes)
            obs2 = aut.get_last_token_weights(word, suffixes)
            if not self._comparator.next_tokens_equivalent_output(obs1, obs2):
                errorCount += 1
                if counterexample is None:
                    counterexample = word 
                if not self._compute_epsilon_star:
                    return False, counterexample               
        if errorCount > 0:
            self._calculate_epsilon_star_with(errorCount)

        return counterexample is None, counterexample

    @property
    def alphabet(self):
        return self._target_model.alphabet

    @property
    def terminal_symbol(self):
        return self._target_model.terminal_symbol

    def _calculate_sample_size(self):
        numberOfCalls = self.equivalence_queries_count
        sample_size = ceil((log(2) * (numberOfCalls + 1) - log(self._delta)) / self._epsilon)
        self.last_sample_size = sample_size
        return sample_size

    def _calculate_epsilon_star_with(self, errorCount: int):
        combinations = comb(self.last_sample_size, errorCount)
        if (self.last_sample_size - errorCount) == 0:
            self.epsilon_star = float('inf')
        else:
            self.epsilon_star = (log(combinations) - log(self._delta)) / (self.last_sample_size - errorCount)

