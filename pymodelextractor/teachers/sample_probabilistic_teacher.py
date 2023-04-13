from pythautomata.base_types.sequence import Sequence
from pythautomata.automata.wheighted_automaton_definition.weighted_automaton import WeightedAutomaton
from pythautomata.abstract.probabilistic_model import ProbabilisticModel
from pymodelextractor.teachers.probabilistic_teacher import ProbabilisticTeacher
from pythautomata.utilities.sequence_generator import SequenceGenerator
from pythautomata.utilities.uniform_length_sequence_generator import UniformLengthSequenceGenerator
from pythautomata.abstract.finite_automaton import FiniteAutomataComparator
from pymodelextractor.utils.data_loader import DataLoader

from typing import Union, Sized


class SampleProbabilisticTeacher(ProbabilisticTeacher):
    def __init__(self, model: ProbabilisticModel, comparator: FiniteAutomataComparator, sample_size: float = None,
                 sequence_generator: SequenceGenerator = None, max_seq_length: int = 128, full_prefix_set = False, parallel_cache = False, max_query_elements = 1_000_000, batch_size = 10_000, cache_from_dataloader:DataLoader = None):
        super().__init__(model, parallel_cache, max_query_elements, batch_size, cache_from_dataloader)
        self._sample_size = sample_size
        self._target_model = model
        self._comparator = comparator
        self._full_prefix_set = full_prefix_set
        if full_prefix_set and sample_size is not None:
            raise ValueError('full_prefix_set value should be False if sample_size is_set')
        
        if sequence_generator is None:
            self._sequence_generator = UniformLengthSequenceGenerator(self._target_model.alphabet, max_seq_length= max_seq_length)
        else:
            self._sequence_generator = sequence_generator
        self.__rand_words = None

    def log_sequence_weight(self, sequence: Sequence):
        return self._target_model.log_sequence_weight(sequence)

    def get_log_probability_error(self, seq, aut: WeightedAutomaton):
        return abs(aut.log_sequence_weight(seq) - self.log_sequence_weight(seq))
    
    def generate_words(self):
        rand_words = None
        if self._full_prefix_set:
            if self.__rand_words is None:
                self.__rand_words = sorted(self._sequence_generator.generate_all_words_up_to_max_length())
            rand_words = self.__rand_words
        else:  
            rand_words = sorted(self._sequence_generator.generate_words(self._sample_size))
        return rand_words

    def equivalence_query(self, aut: WeightedAutomaton) -> Union[tuple[bool, Sized], tuple[bool, None]]:
        self._equivalence_queries_count += 1
        tried = set()
        suffixes = list()
        suffixes.append(self.terminal_symbol)
        for symbol in self.alphabet.symbols:
            suffixes.append(Sequence([symbol]))
        rand_words = self.generate_words()
        for word in rand_words:
            prefixes = sorted(word.get_prefixes(), key=len)
            for prefix in prefixes:
                if prefix not in tried:
                    obs1 = self.last_token_weights(prefix, suffixes)
                    obs2 = aut.get_last_token_weights(prefix, suffixes)
                    if not self._comparator.next_tokens_equivalent_output(obs1, obs2):
                        return False, prefix
                    tried.add(prefix)        
        return True, None

    @property
    def alphabet(self):
        return self._target_model.alphabet

    @property
    def terminal_symbol(self):
        return self._target_model.terminal_symbol
