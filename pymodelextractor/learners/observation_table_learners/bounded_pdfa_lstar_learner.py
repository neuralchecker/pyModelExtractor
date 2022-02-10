import time

from pythautomata.automata.wheighted_automaton_definition.weighted_automaton import WeightedAutomaton
from pythautomata.base_types.sequence import Sequence
from pymodelextractor.learners.observation_table_learners.pdfa_observation_table import PDFAObservationTable, \
    epsilon
from pymodelextractor.learners.observation_table_learners.translators.pdfa_lstar_observation_table_translator import \
     PDFALStarObservationTableTranslation
from pymodelextractor.learners.observation_table_learners.pdfa_lstar_learner import PDFALStarLearner
from pymodelextractor.teachers.probabilistic_teacher import ProbabilisticTeacher
from pymodelextractor.learners.learning_result import LearningResult
from pymodelextractor.exceptions.query_length_exceeded_exception import QueryLengthExceededException
from pymodelextractor.exceptions.number_of_states_exceeded_exception import NumberOfStatesExceededException

class BoundedPDFALStarLearner(PDFALStarLearner):

    def __init__(self, max_states, max_query_length):
        super().__init__()
        self._max_states = max_states
        self._max_query_length = max_query_length
        self._exceeded_max_states = False
        self._exceeded_max_mq_length = False
        self._history = []


    def learn(self, teacher: ProbabilisticTeacher, tolerance, verbose: bool = False) -> LearningResult:
        try:
            result = super().learn(teacher, tolerance)            
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


    def reset(self):
        super().reset()
        self._history = []
   

    def _get_filled_row_for(self, sequence: Sequence) -> list:
        largestSuffixLength = max((len(sequence.value)
                                   for sequence in self.observation_table.get_suffixes()))
        if len(sequence) + largestSuffixLength > self._max_query_length:
            raise QueryLengthExceededException
        return super()._get_filled_row_for(sequence)

    
    def _fill_hole_for(self, sequence: Sequence, suffix):
        suffix = self.observation_table.get_suffixes()[-1]
        if len(sequence) + len(suffix) > self._max_query_length:
            raise QueryLengthExceededException
        super()._fill_hole_for(sequence, suffix)


    def perform_equivalence_query(self, model):
        self._history.append(model)
        if len(model.weighted_states) > self._max_states:
            raise NumberOfStatesExceededException
        return super().perform_equivalence_query(model)

