from pythautomata.base_types.sequence import Sequence
from pymodelextractor.learners.observation_table_learners.pdfa_lstar_learner import PDFALStarLearner
from pymodelextractor.teachers.probabilistic_teacher import ProbabilisticTeacher
from pymodelextractor.learners.learning_result import LearningResult
from pymodelextractor.exceptions.query_length_exceeded_exception import QueryLengthExceededException
from pymodelextractor.exceptions.number_of_states_exceeded_exception import NumberOfStatesExceededException
from pymodelextractor.utils.time_bound_utilities import timeout


class BoundedPDFALStarLearner(PDFALStarLearner):

    def __init__(self, comparator, max_states, max_query_length, max_seconds_run=None):
        super().__init__(comparator=comparator)
        self._max_states = max_states
        self._max_query_length = max_query_length
        self._max_seconds_run = max_seconds_run
        self._exceeded_max_states = False
        self._exceeded_max_mq_length = False        
        self._exceded_time_bound = False    
        self._history = []

    def run_learning_with_time_bound(self, teacher, verbose):        
        try:
            with timeout(self._max_seconds_run):
                super().learn(teacher, verbose) 
        except TimeoutError:
            print("Time Bound Reached") 
            self._exceded_time_bound = True

    def learn(self, teacher: ProbabilisticTeacher, verbose: bool = False) -> LearningResult:
        try:
            if self._max_seconds_run is not None:
                self.run_learning_with_time_bound(teacher, verbose)
            else:
                super().learn(teacher, verbose)
        except NumberOfStatesExceededException:
            print("NumberOfStatesExceeded")
            self._exceeded_max_states = True
        except QueryLengthExceededException:
            print("QueryLengthExceeded")
            self._exceeded_max_mq_length = True
        result = self._learning_results_for(self._history[-1] if len(self._history) > 0 else None)
        result.info['NumberOfStatesExceeded'] = self._exceeded_max_states
        result.info['QueryLengthExceeded'] = self._exceeded_max_mq_length              
        result.info['TimeExceeded'] = self._exceded_time_bound
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

