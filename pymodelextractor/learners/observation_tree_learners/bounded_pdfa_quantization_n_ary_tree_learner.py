from typing import Tuple

from pythautomata.automata.wheighted_automaton_definition.probabilistic_deterministic_finite_automaton import \
    ProbabilisticDeterministicFiniteAutomaton

from pymodelextractor.teachers.probabilistic_teacher import ProbabilisticTeacher
from pymodelextractor.learners.observation_tree_learners.pdfa_quantization_n_ary_tree_learner import \
     PDFAQuantizationNAryTreeLearner
from pymodelextractor.learners.learning_result import LearningResult
from pymodelextractor.exceptions.query_length_exceeded_exception import QueryLengthExceededException
from pymodelextractor.exceptions.number_of_states_exceeded_exception import NumberOfStatesExceededException
from pymodelextractor.utils.time_bound_utilities import timeout


class BoundedPDFAQuantizationNAryTreeLearner(PDFAQuantizationNAryTreeLearner):
    def __init__(self, max_states, max_query_length, max_seconds_run=None):
        super().__init__()
        self._max_states = max_states
        self._max_query_length = max_query_length
        self._max_seconds_run = max_seconds_run
        self._exceeded_max_states = False
        self._exceeded_max_mq_length = False
        self._exceded_time_bound = False
        self._history = []

    def _perform_equivalence_query(self, model):
        self._history.append(model)
        if len(model.weighted_states) > self._max_states:
            raise NumberOfStatesExceededException
        return super()._perform_equivalence_query(model)

    def run_learning_with_time_bound(self, teacher, partitions, verbose):
        try:
            with timeout(self._max_seconds_run):
                super().learn(teacher, partitions, verbose)
        except TimeoutError:
            print("Time Bound Reached")
            self._exceded_time_bound = True

    def learn(self, teacher: ProbabilisticTeacher, partitions: int, verbose: bool = False) -> LearningResult:
        try:
            if self._max_seconds_run is not None:
                self.run_learning_with_time_bound(teacher, partitions, verbose)
            else:
                super().learn(teacher, partitions, verbose)
        except NumberOfStatesExceededException:
            print("NumberOfStatesExceeded")
            self._exceeded_max_states = True
        except QueryLengthExceededException:
            print("QueryLengthExceeded")
            self._exceeded_max_mq_length = True
        hist = list(self._history)
        result = self._learning_results_for(hist[-1] if len(hist) > 0 else None)
        result.info['NumberOfStatesExceeded'] = self._exceeded_max_states
        result.info['QueryLengthExceeded'] = self._exceeded_max_mq_length
        result.info['TimeExceeded'] = self._exceded_time_bound
        return result

    def _learning_results_for(self, model):
        if self._max_seconds_run is not None:
            numberOfStates = len(model.weighted_states) if model is not None else 0
            for count, state in enumerate(model.weighted_states):
                state.name = 'q' + str(count)

            info = {
                'equivalence_queries_count': None,
                'last_token_weight_queries_count': None,
                'observation_tree': None
            }
            return LearningResult(model, numberOfStates, info)
        else:
            return super()._learning_results_for(model)

    def initialization(self, verbose) -> tuple[bool, ProbabilisticDeterministicFiniteAutomaton]:
        ret = super().initialization(verbose)
        if not ret[0]:
            self._tree.max_query_length = self._max_query_length
        return ret
