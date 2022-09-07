import unittest
from pythautomata.automata.wheighted_automaton_definition.probabilistic_deterministic_finite_automaton \
     import ProbabilisticDeterministicFiniteAutomaton
from pythautomata.automata.wheighted_automaton_definition.weighted_state import WeightedState

from pymodelextractor.learners.observation_tree_learners.pdfa_quantization_n_ary_tree_learner \
     import PDFAQuantizationNAryTreeLearner
from pythautomata.model_comparators.wfa_tolerance_comparison_strategy import WFAToleranceComparator
from pythautomata.model_comparators.wfa_quantization_comparison_strategy import WFAQuantizationComparator

from pymodelextractor.teachers.pdfa_teacher import PDFATeacher

from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.symbol import SymbolStr

binaryAlphabet = Alphabet(frozenset((SymbolStr('0'), SymbolStr('1'))))


class TestPDFAQuantizantionNAryTreeLearnerRunningExample(unittest.TestCase):

    def setUp(self):
        self.partitions = 10
        self.comparator = WFAQuantizationComparator(self.partitions)
        self.learner = PDFAQuantizationNAryTreeLearner(self.comparator)

    def generate_running_example(self):
        qlambda = WeightedState("qlambda", 1, 0.0)
        q0 = WeightedState("q0", 0, 0.1)
        q1 = WeightedState("q1", 0, 0.1)
        q01 = WeightedState("q01", 0, 0.1)

        zero = SymbolStr('0')
        one = SymbolStr('1')
        qlambda.add_transition(zero, q0, 0.5)
        qlambda.add_transition(one, q1, 0.5)

        q0.add_transition(zero, q0, 0.3)
        q0.add_transition(one, q01, 0.6)

        q1.add_transition(zero, q01, 0.6)
        q1.add_transition(one, q1, 0.3)

        q01.add_transition(zero, q0, 0.3)
        q01.add_transition(one, q1, 0.6)

        states = {qlambda, q0, q1, q01}
        comparator = WFAToleranceComparator()
        return ProbabilisticDeterministicFiniteAutomaton(binaryAlphabet, states, SymbolStr("$"), comparator,
                                                         "running_example_pdfa")

    def test_1(self):
        model = self.generate_running_example()
        teacher = PDFATeacher(model, self.comparator)
        result = self.learner.learn(teacher)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count'] > 0)
        self.assertTrue(result.info['equivalence_queries_count'] > 0)
