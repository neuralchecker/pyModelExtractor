from pythautomata.base_types.sequence import Sequence
from pymodelextractor.teachers.teacher import Teacher
from pymodelextractor.learners.observation_table_learners.lstar_learner import LStarLearner
from pymodelextractor.exceptions.query_length_exceeded_exception import QueryLengthExceededException
from pymodelextractor.exceptions.number_of_states_exceeded_exception import NumberOfStatesExceededException
from pythautomata.abstract.boolean_model import BooleanModel


class BoundedLStarLearner(LStarLearner):

    def __init__(self, max_states, max_mq_length):
        super().__init__()
        self._max_states = max_states
        self._max_mq_length = max_mq_length
        self._exceeded_max_states = False
        self._exceeded_max_mq_length = False
        self._history = []

    def learn(self, teacher: Teacher):
        try:
            return super().learn(teacher)
        except NumberOfStatesExceededException:
            print("NumberOfStatesExceeded")
            self._exceeded_max_states = True
        except QueryLengthExceededException:
            print("QueryLengthExceededException")
            self._exceeded_max_mq_length = True
        return self._learning_results_for(self._history[-1] if len(self._history) > 0 else None, None)

    def _perform_equivalence_query(self, model: BooleanModel) -> bool:
        self._history.append(model)
        if len(model.states) > self._max_states:
            raise NumberOfStatesExceededException
        return super()._perform_equivalence_query(model)

    def _fill_hole_for(self, sequence: Sequence):
        suffix = self._observation_table.exp[-1]
        if len(sequence) + len(suffix) > self._max_mq_length:
            raise QueryLengthExceededException
        super()._fill_hole_for(sequence)

    def _get_filled_row_for(self, sequence: Sequence) -> list:
        largestSuffixLength = max((len(sequence.value)
                                   for sequence in self._observation_table.exp))
        if len(sequence) + largestSuffixLength > self._max_mq_length:
            raise QueryLengthExceededException
        return super()._get_filled_row_for(sequence)
