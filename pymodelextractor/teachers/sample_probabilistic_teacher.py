from pythautomata.base_types.sequence import Sequence
from pythautomata.automata.wheighted_automaton_definition.weighted_automaton import WeightedAutomaton
from pythautomata.abstract.probabilistic_model import ProbabilisticModel
from pymodelextractor.teachers.probabilistic_teacher import ProbabilisticTeacher
from pythautomata.utilities.sequence_generator import SequenceGenerator
from pymodelextractor.utilities import pdfa_utils
from typing import Union

class SampleProbabilisticTeacher(ProbabilisticTeacher):
    def __init__(self, model: ProbabilisticModel, tolerance: float, sample_size: float, sequence_generator: SequenceGenerator = None, max_seq_length: float = 128):
        super().__init__(tolerance)
        self._sample_size = sample_size
        self.__target_model = model
        if sequence_generator is None:
            self._sequence_generator = SequenceGenerator(self.__target_model.alphabet, max_seq_length= max_seq_length)
        else:
            self._sequence_generator = sequence_generator

    def sequence_weight(self, sequence: Sequence):
        return self.__target_model.sequence_weight(sequence)

    def log_sequence_weight(self, sequence: Sequence):
        return self.__target_model.log_sequence_weight(sequence)

    def last_token_weights(self, sequence: Sequence, required_suffixes: list[Sequence]):
        self._last_token_weight_queries_count +=len(required_suffixes)            
        return self.__target_model.get_last_token_weights(sequence, required_suffixes)

    def get_log_probability_error(self, seq, aut: WeightedAutomaton):
        return abs(aut.log_sequence_weight(seq) - self.log_sequence_weight(seq))
    
    def equivalence_query(self, aut: WeightedAutomaton) -> tuple[bool, Union[Sequence,None]]:        
        self._equivalence_queries_count += 1
        tried = set()
        suffixes = list()
        suffixes.append(self.terminal_symbol)
        for symbol in self.alphabet.symbols:
            suffixes.append(Sequence([symbol]))
        rand_words = sorted(self._sequence_generator.generate_words(self._sample_size))
        for word in rand_words:
            prefixes = sorted(word.get_prefixes(), key=len)
            for prefix in prefixes:
                if prefix not in tried:
                    obs1 = self.last_token_weights(prefix, suffixes)
                    obs2 = aut.get_last_token_weights(prefix, suffixes)
                    if not pdfa_utils.are_within_tolerance_limit(obs1, obs2, self.tolerance):
                        return False, prefix
                    tried.add(prefix)        
        return True, None

    @property
    def alphabet(self):
        return self.__target_model.alphabet

    @property
    def terminal_symbol(self):
        return self.__target_model.terminal_symbol
