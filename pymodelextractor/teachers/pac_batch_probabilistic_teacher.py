from pythautomata.base_types.sequence import Sequence
from pythautomata.automata.wheighted_automaton_definition.weighted_automaton import WeightedAutomaton
from pythautomata.abstract.probabilistic_model import ProbabilisticModel
from pythautomata.utilities.sequence_generator import SequenceGenerator
from pymodelextractor.teachers.pac_probabilistic_teacher import PACProbabilisticTeacher
from pythautomata.abstract.finite_automaton import FiniteAutomataComparator
from typing import Union
import numpy as np


class PACBatchProbabilisticTeacher(PACProbabilisticTeacher):

    def __init__(self, model: ProbabilisticModel, epsilon: float, delta: float,
                 comparator: FiniteAutomataComparator, sequence_generator: SequenceGenerator = None,
                 max_seq_length: float = 128, compute_epsilon_star: bool = True):
        super().__init__(model, comparator, epsilon, delta, sequence_generator, max_seq_length, compute_epsilon_star)
        assert (hasattr(model, 'get_last_token_weights_batch'))

    def equivalence_query(self, aut: WeightedAutomaton) -> tuple[bool, Union[Sequence, None]]:
        self._equivalence_queries_count += 1
        sample_size = self._calculate_sample_size()
        errorCount = 0
        counterexample = None
        suffixes = [self.terminal_symbol]

        for symbol in self.alphabet.symbols:
            suffixes.append(Sequence((symbol,)))

        rand_words = self._sequence_generator.generate_words(sample_size)
        np.sort(rand_words)
        results = self._target_model.get_last_token_weights_batch(rand_words, suffixes)
        for i in range(len(rand_words)):
            word = rand_words[i]
            obs1 = results[i]
            obs2 = aut.get_last_token_weights(word, suffixes)
            if not self._comparator.equivalent_output(obs1, obs2):
                errorCount += 1
                if counterexample is None:
                    counterexample = word
                if not self._compute_epsilon_star:
                    return False, counterexample
        if errorCount > 0:
            self._calculate_epsilon_star_with(errorCount)

        return counterexample is None, counterexample
