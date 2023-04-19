from pythautomata.base_types.sequence import Sequence
from pythautomata.automata.wheighted_automaton_definition.weighted_automaton import WeightedAutomaton
from pythautomata.abstract.probabilistic_model import ProbabilisticModel
from pymodelextractor.teachers.probabilistic_teacher import ProbabilisticTeacher
from pythautomata.utilities.sequence_generator import SequenceGenerator
from pythautomata.abstract.finite_automaton import FiniteAutomataComparator
from pymodelextractor.teachers.sample_probabilistic_teacher import SampleProbabilisticTeacher
from pymodelextractor.utils.data_loader import DataLoader
from typing import Union, Sized


class SampleBatchProbabilisticTeacher(SampleProbabilisticTeacher):
    def __init__(self, model: ProbabilisticModel, comparator: FiniteAutomataComparator, sample_size: float = None,
                 sequence_generator: SequenceGenerator = None, max_seq_length: int = 128, full_prefix_set = False, parallel_cache = False, max_query_elements = 1_000_000, batch_size = 10_000, cache_from_dataloader:DataLoader = None):
        super().__init__(model, comparator, sample_size, sequence_generator, max_seq_length, full_prefix_set, parallel_cache, max_query_elements, batch_size, cache_from_dataloader)
        assert (hasattr(model, 'get_last_token_weights_batch'))
        if self._full_prefix_set:
            self._rand_words_generator = self._sequence_generator.generate_all_words()
            self.__rand_words = []
            self._all_rand_words_precomputed = False
    
    def last_token_weights_batch(self, sequences: list[Sequence], required_suffixes: list[Sequence]):
        self._last_token_weight_queries_count += len(sequences)* len(required_suffixes)          
        return self._target_model.get_last_token_weights_batch(sequences, required_suffixes)
 

    def generate_batch_words(self):
        if self._full_prefix_set: 
            pre_computed_rand_words = len(self.__rand_words)
            pre_computed_rand_words_index = 0
            while not self._all_rand_words_precomputed or pre_computed_rand_words_index<pre_computed_rand_words:
                batch = []        
                for i in range(self._batch_size):
                    if pre_computed_rand_words>0 and pre_computed_rand_words_index<pre_computed_rand_words:
                        batch.append(self.__rand_words[pre_computed_rand_words_index])
                        pre_computed_rand_words_index+=1
                    else:
                        if self._all_rand_words_precomputed:
                            break
                        next_element = next(self._rand_words_generator)
                        if len(next_element) > self._sequence_generator._max_seq_length:
                            self._all_rand_words_precomputed = True
                            break
                        self.__rand_words.append(next_element)
                        batch.append(next_element)                
                yield [batch]
        else:
            total = 0
            while total < self._sample_size:
                to_go = total - self._sample_size       
                words_to_generate = min(self.batch_size, to_go)         
                rand_words = sorted(self._sequence_generator.generate_words(words_to_generate))
                total += len(words_to_generate)
                yield [rand_words]

    def equivalence_query(self, aut: WeightedAutomaton) -> Union[tuple[bool, Sized], tuple[bool, None]]:
        self._equivalence_queries_count += 1
        errorCount = 0
        counterexample = None
        suffixes = [self.terminal_symbol]
        for symbol in self.alphabet.symbols:
            suffixes.append(Sequence((symbol,)))
        generator = self.generate_batch_words()
        for rand_words in next(generator):                 
            results = self.last_token_weights_batch(rand_words, suffixes)
            for i in range(len(rand_words)):
                word = rand_words[i]
                obs1 = results[i]
                obs2 = aut.get_last_token_weights(word, suffixes)
                if not self._comparator.next_tokens_equivalent_output(obs1, obs2):
                    errorCount += 1
                    return False, word
        return counterexample is None, counterexample

    
