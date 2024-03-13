import unittest

from pymodelextractor.learners.observation_tree_learners.kearns_vazirani_learner import \
    KearnsVaziraniLearner
from pymodelextractor.teachers.automaton_teacher import \
    DeterministicFiniteAutomatonTeacher as AutomatonTeacher
from pythautomata.automata.deterministic_finite_automaton import \
    DeterministicFiniteAutomaton
from pythautomata.automata_definitions.tomitas_grammars import TomitasGrammars
from pythautomata.model_comparators.hopcroft_karp_comparison_strategy import \
    HopcroftKarpComparisonStrategy as ComparisonStrategy
from pythautomata.automata_definitions.bollig_habermehl_kern_leucker_automata import BolligHabermehlKernLeuckerAutomata
from pythautomata.automata_definitions.omlin_giles_automata import OmlinGilesAutomata
from pythautomata.utilities.nicaud_dfa_generator import generate_dfa
from pythautomata.base_types.symbol import SymbolStr
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.model_comparators.state_prefix_random_walk import StatePrefixRandomWalkComparisonStrategy \
    as StatePrefixRandomWalk
from pymodelextractor.teachers.pac_comparison_strategy import PACComparisonStrategy
from pymodelextractor.teachers.general_teacher import GeneralTeacher
from itertools import chain


class TestKearnsVaziraniLearner(unittest.TestCase):
    def setUp(self):
        self.learner = KearnsVaziraniLearner()

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
    
    def test_against_many_DFAs(self):
        mergedAutomata = list(chain(TomitasGrammars.get_all_automata(),
                                    BolligHabermehlKernLeuckerAutomata.get_all_automata(),
                                    OmlinGilesAutomata.get_all_automata()))
        for automaton in mergedAutomata:
            teacher = self.teacher(automaton)
            result = self.learner.learn(teacher)
            assert ComparisonStrategy().are_equivalent(
                result.model, automaton)
            
    def test_with_100_states_automaton(self):
        dfa = generate_dfa(alphabet= Alphabet(frozenset(map(SymbolStr, ["0", "1", "2"]))), nominal_size=100, seed=17)
        teacher = self.teacher(dfa)
        result = self.learner.learn(teacher)
        assert ComparisonStrategy().are_equivalent(
            result.model, dfa)
        
    def test_automaton_with_PAC(self):
        dfa = generate_dfa(alphabet= Alphabet(frozenset(map(SymbolStr, ["0", "1"]))), nominal_size=10, seed=17)
        teacher = GeneralTeacher(dfa, 
                                 PACComparisonStrategy(dfa.alphabet, 0.005, 0.005, 20))
        result = self.learner.learn(teacher)
        assert ComparisonStrategy().are_equivalent(
            result.model, dfa)
        
    def test_tomitas_with_state_prefix_RW(self):
        grammar4 = TomitasGrammars.get_automaton_4()
        teacher = AutomatonTeacher(grammar4, StatePrefixRandomWalk(1000))
        result = self.learner.learn(teacher)
        assert ComparisonStrategy().are_equivalent(
            result.model, grammar4)
