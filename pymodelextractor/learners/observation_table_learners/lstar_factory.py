from pymodelextractor.learners.observation_table_learners.generic_lstar_learner import GenericLStarLearner
from pymodelextractor.learners.observation_table_learners.translators.fa_observation_table_translator import FAObservationTableTranslator
from pymodelextractor.learners.observation_table_learners.translators.mm_observation_table_translator import MMObservationTableTranslator

class LStarFactory:

    @staticmethod
    def get_dfa_lstar_learner(max_states = None, max_query_length = None, max_time = None) -> GenericLStarLearner:
        return GenericLStarLearner(FAObservationTableTranslator(), max_states, max_query_length, max_time)

    @staticmethod
    def get_moore_machine_lstar_learner(max_states = None, max_query_length = None, max_time = None) -> GenericLStarLearner:
        return GenericLStarLearner(MMObservationTableTranslator(), max_states, max_query_length, max_time)


