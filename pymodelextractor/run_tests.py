from unittest import TestLoader, TestSuite, TextTestRunner

from pymodelextractor.tests.learners_tests.test_lstar_learner import TestLStarLearner
from pymodelextractor.tests.learners_tests.test_lstarcol_learner import TestLStarColLearner
from pymodelextractor.tests.learners_tests.test_kearns_vazirani_learner import TestKearnsVaziraniLearner
from pymodelextractor.tests.learners_tests.test_pdfa_lstar_learner import TestPDFALStarLearner
from pymodelextractor.tests.learners_tests.test_pdfa_lstarcol_learner import TestPDFALStarColLearner
from tests.learners_tests.test_lambda_star_learner import TestLambdaStarLearnerWithEqualityAlgebra
from tests.learners_tests.test_pdfa_teachers_pdfa_lstar import TestPDFATeachersLStar
from tests.learners_tests.test_pdfa_teachers_pdfa_lstar_col import TestPDFATeachersLStarCol
from tests.learners_tests.test_pac_boolean_model_teacher import TestPACBooleanModelTeachers


def run():
    test_classes_to_run = [TestLStarLearner,
                           TestLStarColLearner,
                           TestPDFALStarLearner,
                           TestKearnsVaziraniLearner,
                           TestLStarLearner,
                           TestPDFALStarColLearner,
                           TestPDFATeachersLStar,
                           TestPDFATeachersLStarCol]#,
                           #TestPACBooleanModelTeachers]                      
    loader = TestLoader()
    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
    meta_suite = TestSuite(suites_list)
    TextTestRunner().run(meta_suite)


if __name__ == '__main__':
    run()
