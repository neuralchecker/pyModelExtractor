from unittest import TestLoader, TestSuite, TextTestRunner

from tests.learners_tests.test_lambda_star_learner import TestLambdaStarLearnerWithEqualityAlgebra
from tests.learners_tests.test_star_learner import TestLStarLearner

def run():
    test_classes_to_run = [TestLStarLearner]
    loader = TestLoader()
    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
    meta_suite = TestSuite(suites_list)
    TextTestRunner().run(meta_suite)


if __name__ == '__main__':
    run()
