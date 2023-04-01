from pymodelextractor.learners.observation_table_learners.general_lstar_learner import GeneralLStarLearner
from pymodelextractor.learners.observation_table_learners.translators.\
    fa_observation_table_translator import FAObservationTableTranslator
from pymodelextractor.learners.observation_table_learners.translators.\
    mm_observation_table_translator import MMObservationTableTranslator
from pymodelextractor.learners.observation_table_learners.translators.partial_dfa_translator\
      import PartialDFATranslator

class LStarFactory:

    @staticmethod
    def get_dfa_lstar_learner(max_states = -1, max_query_length = -1, max_time = -1)\
            -> GeneralLStarLearner:
        return GeneralLStarLearner(FAObservationTableTranslator(), max_states, max_query_length, 
                                   max_time)
    
    @staticmethod
    def get_partial_dfa_lstar_learner(max_states = -1, max_query_length = -1, max_time = -1) \
            -> GeneralLStarLearner:
        return GeneralLStarLearner(PartialDFATranslator(), max_states, max_query_length, max_time)

    @staticmethod
    def get_moore_machine_lstar_learner(max_states = -1, max_query_length = -1, max_time = -1) \
            -> GeneralLStarLearner:
        return GeneralLStarLearner(MMObservationTableTranslator(), max_states, max_query_length,
                                   max_time)


