from typing import Tuple

from pythautomata.automata.wheighted_automaton_definition.probabilistic_deterministic_finite_automaton import \
    ProbabilisticDeterministicFiniteAutomaton as PDFA, ProbabilisticDeterministicFiniteAutomaton
from pythautomata.model_comparators.wfa_tolerance_comparison_strategy import WFAToleranceComparator
from pymodelextractor.teachers.probabilistic_teacher import ProbabilisticTeacher
from pymodelextractor.learners.observation_tree_learners.pdfa_quantization_n_ary_tree_learner import \
     PDFAQuantizationNAryTreeLearner
from pymodelextractor.learners.learning_result import LearningResult
from pymodelextractor.exceptions.query_length_exceeded_exception import QueryLengthExceededException
from pymodelextractor.exceptions.number_of_states_exceeded_exception import NumberOfStatesExceededException
from pymodelextractor.utils.time_bound_utilities import timeout
from pymodelextractor.learners.observation_table_learners.observation_table import epsilon
from pythautomata.automata.wheighted_automaton_definition.weighted_state import WeightedState

class BoundedPDFAQuantizationNAryTreeLearner(PDFAQuantizationNAryTreeLearner):
    def __init__(self, partitioner, max_states, max_query_length, max_seconds_run=None, generate_partial_hipothesis = False, pre_cache_queries_for_building_hipothesis = False, check_probabilistic_hipothesis = True):
        super().__init__(partitioner, pre_cache_queries_for_building_hipothesis, check_probabilistic_hipothesis)
        self._max_states = max_states
        self._max_query_length = max_query_length
        self._max_seconds_run = max_seconds_run
        self._exceeded_max_states = False
        self._exceeded_max_mq_length = False
        self._exceded_time_bound = False
        self._history = []        
        self._generate_partial_hipothesis = generate_partial_hipothesis
        

    def _perform_equivalence_query(self, model):
        self._history.append(model)
        if len(model.weighted_states) > self._max_states:
            raise NumberOfStatesExceededException
        return super()._perform_equivalence_query(model)

    def run_learning_with_time_bound(self, teacher, verbose):
        try:
            with timeout(self._max_seconds_run):
                super().learn(teacher, verbose)
        except TimeoutError:
            if verbose: print("Time Bound Reached")
            self._exceded_time_bound = True

    def learn(self, teacher: ProbabilisticTeacher, verbose: bool = False) -> LearningResult:
        try:
            if self._max_seconds_run is not None:
                self.run_learning_with_time_bound(teacher, verbose)
            else:
                super().learn(teacher, verbose)
        except NumberOfStatesExceededException:
            if verbose: print("NumberOfStatesExceeded")
            self._exceeded_max_states = True
        except QueryLengthExceededException:
            if verbose: print("QueryLengthExceeded")
            self._exceeded_max_mq_length = True        

        if not self._exceeded_max_states and self._generate_partial_hipothesis and (self._exceeded_max_mq_length or self._exceded_time_bound) and len(self._tree.leaves)>0:         
            partial_hipothesis = self.partial_hipothesis()
            self._history.append(partial_hipothesis)

        hist = list(self._history)
        result = self._learning_results_for(hist[-1] if len(hist) > 0 else None)
        result.info['NumberOfStatesExceeded'] = self._exceeded_max_states
        result.info['QueryLengthExceeded'] = self._exceeded_max_mq_length
        result.info['TimeExceeded'] = self._exceded_time_bound
        return result

    def initialization(self, verbose) -> tuple[bool, ProbabilisticDeterministicFiniteAutomaton]:
        try:
            ret = super().initialization(verbose)
            if not ret[0]:
                self._tree.max_query_length = self._max_query_length
            return ret
        except QueryLengthExceededException:
            print("QueryLengthExceeded")
            self._exceeded_max_mq_length = True
            return True, None

    def partial_hipothesis(self) -> PDFA:        
        if self._pre_cache_queries_for_building_hipothesis:
                self._tree.cache_queries_for_building_hipothesis()        
        states = {}
        symbols = list(self._alphabet.symbols)
        symbols.sort()
        for leaf_str, leaf in self._tree.leaves.items():
            initial_weight = 1 if leaf_str == epsilon else 0
            terminal_symbol_probability = leaf.probabilities[self.terminal_symbol]
            state = WeightedState(leaf_str, initial_weight, terminal_symbol_probability)
            states[leaf_str] = state
        unknown_state =  WeightedState(self._tree.unknown_leaf, 0, -1)     
        states[self._tree.unknown_leaf] = unknown_state
        accessed_states = set()
        for access_string, state in states.items():
            if access_string!=self._tree.unknown_leaf:
                for symbol in symbols:
                    access_string_of_transition, _ = self._tree.sift(access_string + symbol, update=False)
                    if access_string_of_transition != access_string:
                        accessed_states.add(access_string_of_transition)
                    state.add_transition(symbol, states[access_string_of_transition],
                                            self._tree.leaves[access_string].probabilities[symbol])        
        
        for symbol in symbols:
            states[self._tree.unknown_leaf].add_transition(symbol, states[self._tree.unknown_leaf], -1)

        for state in list(states.keys()).copy():
            if state not in accessed_states and states[state].initial_weight != 1:
                del states[state]

        comparator = WFAToleranceComparator()
        states = set(states.values())
        return PDFA(self._alphabet, states, self.terminal_symbol, comparator=comparator, check_is_probabilistic = False)