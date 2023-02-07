from pythautomata.base_types.sequence import Sequence
from pythautomata.automata.wheighted_automaton_definition.weighted_automaton import WeightedAutomaton
from pythautomata.abstract.probabilistic_model import ProbabilisticModel
from pymodelextractor.teachers.probabilistic_teacher import ProbabilisticTeacher
from pythautomata.utilities.sequence_generator import SequenceGenerator
from pythautomata.abstract.finite_automaton import FiniteAutomataComparator
from pymodelextractor.teachers.sample_probabilistic_teacher import SampleProbabilisticTeacher
from typing import Union, Sized


class SampleBatchProbabilisticTeacher(SampleProbabilisticTeacher):
    def __init__(self, model: ProbabilisticModel, comparator: FiniteAutomataComparator, sample_size: float = None,
                 sequence_generator: SequenceGenerator = None, max_seq_length: int = 128, full_prefix_set = False):
        super().__init__(model, comparator, sample_size, sequence_generator, max_seq_length, full_prefix_set)
        assert (hasattr(model, 'get_last_token_weights_batch'))
    
    def last_token_weights_batch(self, sequences: list[Sequence], required_suffixes: list[Sequence]):
        self._last_token_weight_queries_count += len(sequences)* len(required_suffixes)          
        return self._target_model.get_last_token_weights_batch(sequences, required_suffixes)

    def equivalence_query(self, aut: WeightedAutomaton) -> Union[tuple[bool, Sized], tuple[bool, None]]:
        self._equivalence_queries_count += 1
        errorCount = 0
        counterexample = None
        suffixes = [self.terminal_symbol]
        for symbol in self.alphabet.symbols:
            suffixes.append(Sequence((symbol,)))

        rand_words = self.generate_words()
        results = self.last_token_weights_batch(rand_words, suffixes)
        for i in range(len(rand_words)):
            word = rand_words[i]
            obs1 = results[i]
            obs2 = aut.get_last_token_weights(word, suffixes)
            if not self._comparator.equivalent_output(obs1, obs2):
                errorCount += 1
                return False, word
        return counterexample is None, counterexample

    
