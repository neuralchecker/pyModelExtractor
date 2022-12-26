import unittest

from pythautomata.automata_definitions.weighted_tomitas_grammars import WeightedTomitasGrammars

from pymodelextractor.learners.observation_table_learners.pdfa_lstar_learner import PDFALStarLearner
from pythautomata.model_comparators.wfa_tolerance_comparison_strategy import WFAToleranceComparator as PDFAComparator

from pymodelextractor.teachers.pdfa_teacher import PDFATeacher
from pythautomata.base_types.symbol import SymbolStr
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.automata.wheighted_automaton_definition.probabilistic_deterministic_finite_automaton import \
    ProbabilisticDeterministicFiniteAutomaton
from pythautomata.automata.wheighted_automaton_definition.weighted_state import WeightedState
from pythautomata.model_comparators.wfa_tolerance_comparison_strategy import WFAToleranceComparator

binaryAlphabet = Alphabet(frozenset((SymbolStr('a'), SymbolStr('b'))))
a = binaryAlphabet['a']
b = binaryAlphabet['b']

class TestPDFALStarLearnerWTolerance(unittest.TestCase):
       

    def create_PDFA(self):
        q0 = WeightedState("q0", 1, 0.1)
        q1 = WeightedState("q1", 0, 0.1)
        q2 = WeightedState("q2", 0, 0.1)
        

        q0.add_transition(a, q1, 0.4)
        q0.add_transition(b, q2, 0.5)
        q1.add_transition(a, q0, 0.5)
        q1.add_transition(b, q2, 0.4)
        q2.add_transition(a, q1, 0.48)
        q2.add_transition(b, q2, 0.42)        

        states = {q0, q1, q2}
        comparator = WFAToleranceComparator()
        return ProbabilisticDeterministicFiniteAutomaton(binaryAlphabet, states, SymbolStr("$"), comparator, "SamplePDFA")

    def test_1(self):
        comparator = PDFAComparator(0.099999)
        learner =  PDFALStarLearner(comparator)
        model = self.create_PDFA()
        teacher = PDFATeacher(model, comparator)
        result = learner.learn(teacher)
        extracted_model = result.model
        extracted_model.export("./")
        self.assertTrue(comparator.are_equivalent(model, extracted_model))        