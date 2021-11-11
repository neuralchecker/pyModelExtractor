from unittest import TestLoader, TestSuite, TextTestRunner

from pymodelextractor.tests.learners_tests.test_lambda_star_learner import TestLambdaStarLearnerWithEqualityAlgebra
from pymodelextractor.tests.learners_tests.test_lstar_learner import TestLStarLearner
from pymodelextractor.tests.learners_tests.test_kearns_vazirani_learner import TestKearnsVaziraniLearner

def run():
    test_classes_to_run = [TestLStarLearner, TestKearnsVaziraniLearner]
    loader = TestLoader()
    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
    meta_suite = TestSuite(suites_list)
    TextTestRunner().run(meta_suite)


if __name__ == '__main__':
    run()
