from unittest import TestLoader, TestSuite, TextTestRunner

from pymodelextractor.tests.learners_tests.test_lstar_learner import TestLStarLearner
from pymodelextractor.tests.learners_tests.test_bounded_lstar_learner import TestBoundedLStarLearner
from pymodelextractor.tests.learners_tests.test_lstarcol_learner import TestLStarColLearner
from pymodelextractor.tests.learners_tests.test_kearns_vazirani_learner import TestKearnsVaziraniLearner
from pymodelextractor.tests.learners_tests.test_pdfa_lstar_learner import TestPDFALStarLearner
from pymodelextractor.tests.learners_tests.test_pdfa_lstarcol_learner import TestPDFALStarColLearner
from pymodelextractor.tests.learners_tests.test_pdfa_kearns_vazirani_learner import TestPDFAKearnsVaziraniLearner
from tests.learners_tests.test_lambda_star_learner import TestLambdaStarLearnerWithEqualityAlgebra
from tests.learners_tests.test_pdfa_teachers_pdfa_lstar import TestPDFATeachersLStar
from tests.learners_tests.test_pdfa_teachers_pdfa_lstar_col import TestPDFATeachersLStarCol
from tests.learners_tests.test_pac_boolean_model_teacher import TestPACBooleanModelTeachers
from tests.learners_tests.test_pdfa_quantization_n_ary_tree_learner import TestPDFAQuantizantionNAryTreeLearner
from tests.learners_tests.test_pdfa_quantization_n_ary_tree_learner_metrics import TestPDFAQuantizantionNAryTreeLearnerMetrics
from tests.learners_tests.test_bounded_pdfa_quantization_n_ary_tree_learner import TestBoundedPDFAQuantizantionNAryTreeLearner
from tests.learners_tests.test_bounded_pdfa_lstar_learner import TestBoundedPDFALStarLearner
from tests.learners_tests.test_pac_batch_teacher_quant import TestPACBatchTeacherQuant
from tests.learners_tests.test_pdfa_quantization_n_ary_tree_learner_running_example import TestPDFAQuantizantionNAryTreeLearnerRunningExample

def run():
    test_classes_to_run = [TestLStarLearner,
                           TestLStarColLearner,
                           TestPDFALStarLearner,
                           TestKearnsVaziraniLearner,
                           TestLStarLearner,
                           TestPDFALStarColLearner,
                           TestPDFATeachersLStar,
                           TestPDFATeachersLStarCol,
                           TestPACBooleanModelTeachers, 
                           TestBoundedPDFAQuantizantionNAryTreeLearner, 
                           TestPDFAQuantizantionNAryTreeLearner,
                           TestBoundedPDFALStarLearner ]             
    test_classes_to_run = [ TestPDFAQuantizantionNAryTreeLearnerRunningExample ]               
    loader = TestLoader()
    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
    meta_suite = TestSuite(suites_list)
    TextTestRunner().run(meta_suite)


if __name__ == '__main__':
    run()
