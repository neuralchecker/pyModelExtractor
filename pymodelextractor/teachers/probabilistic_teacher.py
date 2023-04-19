from abc import ABC, abstractmethod
from collections import OrderedDict

from pythautomata.base_types.symbol import Symbol
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.sequence import Sequence
from pythautomata.automata.wheighted_automaton_definition.weighted_automaton import WeightedAutomaton
from pymodelextractor.utils.data_loader import DataLoader
from pythautomata.abstract.probabilistic_model import ProbabilisticModel


from collections import OrderedDict
from multiprocessing import Process, Manager
from typing import Union


class ProbabilisticTeacher(ABC):

    def __init__(self, model: ProbabilisticModel, parallel_cache = False, max_query_elements = 1_000_000, batch_size = 10_000, cache_from_dataloader:DataLoader = None):
        self._equivalence_queries_count = 0
        self._last_token_weight_queries_count = 0        
        self._target_model = model
        self._parallel_cache = parallel_cache
        self._max_query_elements = max_query_elements   
        self._batch_size = batch_size
        if self._parallel_cache:
            manager = Manager()
            self._cache = manager.dict()
            self._job = Process(target=self.fill_cache, args=(self._cache, model,self._max_query_elements, batch_size)) 
            self._job.start() 
        if cache_from_dataloader is not None:
            if not self._parallel_cache:
                self._cache = dict()
            self._cache.update(cache_from_dataloader.get_data())

    def sequence_weight(self, sequence: Sequence):
        return self._target_model.sequence_weight(sequence)

    @abstractmethod
    def log_sequence_weight(self, sequence: Sequence) -> float:
        raise NotImplementedError

    def last_token_weights(self, sequence: Sequence, required_suffixes: list[Sequence]):
        self._last_token_weight_queries_count +=len(required_suffixes)            
        return self._target_model.get_last_token_weights(sequence, required_suffixes)

    def equivalence_query(self, aut: WeightedAutomaton) -> tuple[bool, Union[Sequence, None]]:
        raise NotImplementedError

    def next_token_probabilities(self, sequence: Sequence) -> OrderedDict[Symbol, float]:
        if self._parallel_cache:            
            if sequence in self._cache:
                return self._cache[sequence]
        symbols = list(self.alphabet.symbols)
        symbols.sort()
        symbols = [self.terminal_symbol] + symbols
        probabilities = self.last_token_weights(sequence, symbols)
        probabilities = OrderedDict(zip(symbols, probabilities))
        return probabilities

    def reset(self) -> None:
        self._equivalence_queries_count = 0
        self._last_token_weight_queries_count = 0

    def log_probability_error(self, seq, aut: WeightedAutomaton) -> float:
        return abs(aut.log_sequence_weight(seq) - self.log_sequence_weight(seq))

    @property
    @abstractmethod
    def alphabet(self) -> Alphabet:
        raise NotImplementedError

    @property
    @abstractmethod
    def terminal_symbol(self) -> Symbol:
        raise NotImplementedError

    @property
    def equivalence_queries_count(self):
        return self._equivalence_queries_count

    @property
    def last_token_weight_queries_count(self):
        return self._last_token_weight_queries_count

    def next_token_probabilities_batch(self, sequences):
        assert hasattr(self._target_model, "get_last_token_weights_batch")
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
    
    def fill_cache(self, cache, model, max_query_elements, batch_size):
        total_elements = 0
        generator = self._sequence_generator.generate_all_words()
        symbols = list(self.alphabet.symbols)
        symbols.sort()
        symbols = [self.terminal_symbol] + symbols
        use_batch = hasattr(self._target_model, "get_last_token_weights_batch")
        while total_elements<max_query_elements:
            queries = []
            if use_batch: 
                for _ in range(batch_size):
                    queries.append(next(generator))                                 
                results = model.get_last_token_weights_batch(queries, symbols)     
            else:
                queries = [next(generator)]
                results = [self.last_token_weights(queries[0], symbols)]                
            results_od = [OrderedDict(zip(symbols, x)) for x in results]
            final_results  = dict(zip(queries, results_od))            
            cache.update(final_results)

    def __del__(self):
        if hasattr(self, "_parallel_cache") and self._parallel_cache:
            self._job.terminate()
