import unittest

from pymodelextractor.learners.observation_table_learners.bounded_lstar_learner import \
    BoundedLStarLearner
from pymodelextractor.teachers.automaton_teacher import \
    DeterministicFiniteAutomatonTeacher as AutomatonTeacher
from pythautomata.automata.deterministic_finite_automaton import \
    DeterministicFiniteAutomaton
from pythautomata.automata_definitions.tomitas_grammars import TomitasGrammars
from pythautomata.automata_definitions.bollig_habermehl_kern_leucker_automata import BolligHabermehlKernLeuckerAutomata
from pythautomata.automata_definitions.omlin_giles_automata import OmlinGilesAutomata
from pythautomata.model_comparators.hopcroft_karp_comparison_strategy import \
    HopcroftKarpComparisonStrategy as ComparisonStrategy
from itertools import chain
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.symbol import SymbolStr
from pythautomata.utilities import abbadingo_one_dfa_generator

binaryAlphabet = Alphabet(frozenset((SymbolStr('0'), SymbolStr('1'))))


class TestBoundedLStarLearner(unittest.TestCase):
    def setUp(self):
        self.learner = BoundedLStarLearner(max_mq_length=10, max_states=10)

    def teacher(self, automaton: DeterministicFiniteAutomaton) -> AutomatonTeacher:
        return AutomatonTeacher(automaton, ComparisonStrategy())

    def test_tomitas_1(self):
        grammar1 = TomitasGrammars.get_automaton_1()
        teacher = self.teacher(grammar1)
        result = self.learner.learn(teacher)
        assert ComparisonStrategy().are_equivalent(
            result.model, grammar1)

    def test_tomitas_2(self):
        grammar2 = TomitasGrammars.get_automaton_2()
        teacher = self.teacher(grammar2)
        result = self.learner.learn(teacher)
        assert ComparisonStrategy().are_equivalent(
            result.model, grammar2)

    def test_tomitas_3(self):
        grammar3 = TomitasGrammars.get_automaton_3()
        teacher = self.teacher(grammar3)
        result = self.learner.learn(teacher)
        assert ComparisonStrategy().are_equivalent(
            result.model, grammar3)

    def test_tomitas_4(self):
        grammar4 = TomitasGrammars.get_automaton_4()
        teacher = self.teacher(grammar4)
        result = self.learner.learn(teacher)
        assert ComparisonStrategy().are_equivalent(
            result.model, grammar4)

    def test_big_DFA(self):
        dfa = abbadingo_one_dfa_generator.generate_dfa(alphabet=binaryAlphabet, nominal_size=30, seed=10)
        teacher = self.teacher(dfa)
        result = self.learner.learn(teacher)
        self.assertFalse(ComparisonStrategy().are_equivalent(
            result.model, dfa))

    def test_against_many_DFAs(self):
        mergedAutomata = list(chain(TomitasGrammars.get_all_automata(),
                                    BolligHabermehlKernLeuckerAutomata.get_all_automata(),
                                    OmlinGilesAutomata.get_all_automata()))
        for automaton in mergedAutomata:
            teacher = self.teacher(automaton)
            result = self.learner.learn(teacher)
            assert ComparisonStrategy().are_equivalent(
                result.model, automaton)
