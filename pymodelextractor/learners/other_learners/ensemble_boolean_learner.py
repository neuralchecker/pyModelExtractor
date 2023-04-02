from pythautomata.base_types.sequence import Sequence
from pymodelextractor.learners.observation_table_learners.general_observation_table\
      import GeneralObservationTable
from pymodelextractor.learners.learning_result import LearningResult
from pymodelextractor.teachers.general_teacher import GeneralTeacher
import time
from pymodelextractor.utils.time_bound_utilities import timeout
from pythautomata.other_models.composed_boolean_model import ComposedBooleanModel
lamda = Sequence()

no_log = 0
info_log = 1
debug_log = 2
trace_log = 3

class EnsembleBooleanLearner:
    def __init__(self, learning_functions, max_time = -1):
        self._learning_functions = learning_functions
        self._max_time = max_time
        self._last_model = None
        
    def learn(self, teacher) -> LearningResult:
        results = {}
        models = []
        for learning_fun in self._learning_functions:
            result = learning_fun(teacher)
            models.append(result.model)
            results[result.model.name] = result
        composed_model = ComposedBooleanModel(models, models[0].alphabet, lambda *args: sum(args)>(len(args)/2))
        meta_result = LearningResult(composed_model, state_count=-1, info = results)
        return meta_result

