from pymodelextractor.learners.learner import Learner
from pymodelextractor.learners.observation_table_learners.translators.mm_observation_table_translator import MMObservationTableTranslator


class MMLStarLearner(Learner):

    def __init__(self):
        self._model_translator = MMObservationTableTranslator()

    def _build_observation_table(self):
        self._observation_table = MMObservationTableTranslator()

    def _initialize_observation_table(self):
        self

    
    