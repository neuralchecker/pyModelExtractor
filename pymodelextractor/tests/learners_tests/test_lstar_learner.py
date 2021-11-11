import unittest

from pymodelextractor.learners.observation_table_learners.lstar_learner import \
    LStarLearner
from pymodelextractor.teachers.automaton_teacher import \
    DeterministicFiniteAutomatonTeacher as AutomatonTeacher
from pythautomata.automata.deterministic_finite_automaton import \
    DeterministicFiniteAutomaton
from pythautomata.automata_definitions.tomitas_grammars import TomitasGrammars
from pythautomata.automata_definitions.bollig_habermehl_kern_leucker_automata import BolligHabermehlKernLeuckerAutomata
from pythautomata.automata_definitions.omlin_giles_automata import OmlinGilesAutomata
from pythautomata.automata_definitions.other_automata import OtherAutomata
from pythautomata.model_comparators.hopcroft_karp_comparison_strategy import \
    HopcroftKarpComparisonStrategy as ComparisonStrategy
from itertools import chain


class TestLStarLearner(unittest.TestCase):
    def setUp(self):
        self.learner = LStarLearner()

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