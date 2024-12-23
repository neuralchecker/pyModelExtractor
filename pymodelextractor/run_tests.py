from unittest import TestLoader, TestSuite, TextTestRunner

from pymodelextractor.tests.learners_tests.test_lstar_learner import TestLStarLearner
from pymodelextractor.tests.learners_tests.test_bounded_lstar_learner import TestBoundedLStarLearner
from pymodelextractor.tests.learners_tests.test_lstarcol_learner import TestLStarColLearner
from pymodelextractor.tests.learners_tests.test_kearns_vazirani_learner import TestKearnsVaziraniLearner
from pymodelextractor.tests.learners_tests.test_pdfa_lstarcol_learner import TestPDFALStarColLearner
from pymodelextractor.tests.learners_tests.test_pdfa_lstar_quant_learner import TestPDFALStarQuantLearner
from pymodelextractor.tests.learners_tests.test_pdfa_lstar_tolerance_learner import TestPDFALStarToleranceLearner
from pymodelextractor.tests.learners_tests.test_pdfa_lstarcol_quant_learner import TestPDFALStarColQuantLearner
from pymodelextractor.tests.learners_tests.test_lambda_star_learner import TestLambdaStarLearnerWithEqualityAlgebra
from pymodelextractor.tests.learners_tests.test_pdfa_teachers_pdfa_lstar import TestPDFATeachersLStar
from pymodelextractor.tests.learners_tests.test_pdfa_teachers_pdfa_lstar_col import TestPDFATeachersLStarCol
from pymodelextractor.tests.learners_tests.test_pac_boolean_model_teacher import TestPACBooleanModelTeachers
from pymodelextractor.tests.learners_tests.test_pdfa_quantization_n_ary_tree_learner import TestPDFAQuantizantionNAryTreeLearner
from pymodelextractor.tests.learners_tests.test_pdfa_quantization_n_ary_tree_learner_metrics \
     import TestPDFAQuantizantionNAryTreeLearnerMetrics
from pymodelextractor.tests.learners_tests.test_bounded_pdfa_quantization_n_ary_tree_learner \
     import TestBoundedPDFAQuantizantionNAryTreeLearner
from pymodelextractor.tests.learners_tests.test_bounded_pdfa_lstar_learner import TestBoundedPDFALStarLearner
from pymodelextractor.tests.learners_tests.test_pac_batch_teacher_quant import TestPACBatchTeacherQuant
from pymodelextractor.tests.learners_tests.test_pdfa_quantization_n_ary_tree_learner_running_example \
     import TestPDFAQuantizantionNAryTreeLearnerRunningExample
from pymodelextractor.tests.learners_tests.test_mm_lstar_learner import TestMMLStarLearner
from pymodelextractor.tests.learners_tests.test_general_lstar_learner import TestGeneralLStarLearner
from pymodelextractor.tests.learners_tests.test_base_observation_table import TestBaseObservationTable
from pymodelextractor.tests.learners_tests.test_partial_table_translator import TestPartialDFATranslator
from pymodelextractor.tests.learners_tests.test_ensemble_boolean_learner import TestEnsembleBooleanLearner
from pymodelextractor.tests.learners_tests.test_ensemble_probabilistic_learner import TestEnsembleProbabilisticLearner
from pymodelextractor.tests.learners_tests.test_observation_pack_learner import TestObservationPackLearner

def run():
     test_classes_to_run = [TestLStarLearner,
                              TestLStarColLearner,
                              TestPDFALStarQuantLearner,
                              TestPDFALStarToleranceLearner,
                              TestPDFALStarColQuantLearner,
                              TestPDFALStarColLearner,
                              TestKearnsVaziraniLearner,
                              TestLStarLearner,
                              TestPDFATeachersLStar,
                              TestPACBooleanModelTeachers, 
                              TestBoundedPDFAQuantizantionNAryTreeLearner, 
                              TestPDFAQuantizantionNAryTreeLearner,
                              TestBoundedPDFALStarLearner,
                              TestBoundedLStarLearner,
                              TestLambdaStarLearnerWithEqualityAlgebra,
                              TestPDFAQuantizantionNAryTreeLearnerMetrics,
                              TestPACBatchTeacherQuant,
                              TestPDFAQuantizantionNAryTreeLearnerRunningExample,
                              TestMMLStarLearner,
                              TestGeneralLStarLearner,
                              TestPDFATeachersLStarCol,
                              TestBaseObservationTable,
                              TestPartialDFATranslator,
                              TestEnsembleBooleanLearner,
                              TestEnsembleProbabilisticLearner,
                              TestObservationPackLearner]
     
     loader = TestLoader()
     suites_list = []
     for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
     meta_suite = TestSuite(suites_list)
     TextTestRunner().run(meta_suite)


if __name__ == '__main__':
    run()