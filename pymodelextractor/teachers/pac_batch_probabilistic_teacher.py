from pythautomata.base_types.sequence import Sequence
from pythautomata.automata.wheighted_automaton_definition.weighted_automaton import WeightedAutomaton
from pythautomata.abstract.probabilistic_model import ProbabilisticModel
from pythautomata.utilities.sequence_generator import SequenceGenerator
from pymodelextractor.teachers.pac_probabilistic_teacher import PACProbabilisticTeacher
from pythautomata.abstract.finite_automaton import FiniteAutomataComparator
from pythautomata.base_types.symbol import Symbol
from typing import Union
import numpy as np
from collections import OrderedDict
from multiprocessing import Process, Manager

class PACBatchProbabilisticTeacher(PACProbabilisticTeacher):

    def __init__(self, model: ProbabilisticModel, epsilon: float, delta: float,
                 comparator: FiniteAutomataComparator, sequence_generator: SequenceGenerator = None,
                 max_seq_length: float = 128, compute_epsilon_star: bool = True, parallel_cache = False, max_query_elements = 1_000_000, batch_size = 10_000):
        super().__init__(model, comparator, epsilon, delta, sequence_generator, max_seq_length, compute_epsilon_star)
        assert (hasattr(model, 'get_last_token_weights_batch'))
        self._parallel_cache = parallel_cache
        self._max_query_elements = max_query_elements
        if self._parallel_cache:
            manager = Manager()
            self._cache = manager.dict()
            job = Process(target=self.fill_cache, args=(self._cache, model,self._max_query_elements, batch_size)) 
            job.start() 

    def fill_cache(self, cache, model, max_query_elements, batch_size):
        total_elements = 0
        generator = self._sequence_generator.generate_all_words()
        symbols = list(self.alphabet.symbols)
        symbols.sort()
        symbols = [self.terminal_symbol] + symbols

        while total_elements<max_query_elements:
            queries = []
            for _ in range(batch_size):
                queries.append(next(generator))                      
            results = model.get_last_token_weights_batch(queries, symbols)                         
            results_od = [OrderedDict(zip(symbols, x)) for x in results]
            final_results  = dict(zip(queries, results_od))            
            cache.update(final_results)

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
            if not self._comparator.next_tokens_equivalent_output(obs1, obs2):
                errorCount += 1
                if counterexample is None:
                    counterexample = word
                if not self._compute_epsilon_star:
                    return False, counterexample
        if errorCount > 0:
            self._calculate_epsilon_star_with(errorCount)

        return counterexample is None, counterexample

    def next_token_probabilities(self, sequence: Sequence) -> OrderedDict[Symbol, float]:
        if self._parallel_cache:            
            if sequence in self._cache:
                return self._cache[sequence]
        return super().next_token_probabilities(sequence)

    def next_token_probabilities_batch(self, sequences):
        symbols = list(self.alphabet.symbols)
        symbols.sort()
        symbols = [self.terminal_symbol] + symbols

        if self._parallel_cache:
            queries = set()
            results_already_in_cache = dict()
            for sequence in sequences:
                if sequence not in self._cache:
                    queries.add(sequence)
                else:
                    results_already_in_cache[sequence] = self._cache[sequence]
            results = self._target_model.get_last_token_weights_batch(queries, symbols)   
            results_od = [OrderedDict(zip(symbols, x)) for x in results]
            final_results  = dict(zip(queries, results_od))
            self._cache.update(final_results)
            final_results.update(results_already_in_cache)
        else:        
            results = self._target_model.get_last_token_weights_batch(sequences, symbols)
            results_od = [OrderedDict(zip(symbols, x)) for x in results]
            final_results = zip(sequences, results_od)
        return final_results