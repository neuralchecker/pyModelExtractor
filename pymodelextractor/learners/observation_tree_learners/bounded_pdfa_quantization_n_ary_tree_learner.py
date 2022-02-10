from re import I
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.symbol import Symbol
from pythautomata.automata.wheighted_automaton_definition.weighted_state import WeightedState
from pythautomata.model_comparators.wfa_tolerance_comparison_strategy import WFAToleranceComparator
from pythautomata.utilities import pdfa_utils
from pymodelextractor.teachers.probabilistic_teacher import ProbabilisticTeacher
from pymodelextractor.learners.observation_tree_learners.pdfa_quantization_n_ary_tree_learner import PDFAQuantizationNAryTreeLearner
from pythautomata.automata.wheighted_automaton_definition.probabilistic_deterministic_finite_automaton import ProbabilisticDeterministicFiniteAutomaton as PDFA
from pymodelextractor.learners.observation_table_learners.observation_table import epsilon #TODO: Fix this smelly smell https://i.pinimg.com/originals/b9/76/b7/b976b79635bf31c0d97e38297cb54db0.jpg
from pymodelextractor.learners.learning_result import LearningResult
from collections import OrderedDict
import numpy as np
from pymodelextractor.exceptions.query_length_exceeded_exception import QueryLengthExceededException
from pymodelextractor.exceptions.number_of_states_exceeded_exception import NumberOfStatesExceededException

class BoundedPDFAQuantizationNAryTreeLearner(PDFAQuantizationNAryTreeLearner):
    def __init__(self, max_states, max_query_length):     
        super().__init__()
        self._max_states = max_states
        self._max_query_length = max_query_length
        self._exceeded_max_states = False
        self._exceeded_max_mq_length = False
        self._history = []

    def _perform_equivalence_query(self, model):
        self._history.append(model)
        if len(model.weighted_states) > self._max_states:
            raise NumberOfStatesExceededException
        return super()._perform_equivalence_query(model)


    def learn(self, teacher: ProbabilisticTeacher, partitions: int, verbose: bool = False) -> LearningResult:
        try:
            result = super().learn(teacher, partitions)            
            result.info['NumberOfStatesExceeded'] = False
            result.info['QueryLengthExceeded'] = False
            return result
        except NumberOfStatesExceededException:
            print("NumberOfStatesExceeded")
            self._exceeded_max_states = True
        except QueryLengthExceededException:
            print("QueryLengthExceeded")
            self._exceeded_max_mq_length = True
        result = self._learning_results_for(self._history[-1] if len(self._history) > 0 else None)
        if self._exceeded_max_states: 
            result.info['NumberOfStatesExceeded'] = True
        if self._exceeded_max_mq_length: 
            result.info['QueryLengthExceeded'] = True
        return result

    def initialization(self) -> None:
        ret =  super().initialization()
        self._tree.max_query_length = self._max_query_length
        return ret