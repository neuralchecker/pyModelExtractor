import unittest

from pymodelextractor.learners.observation_table_learners.mm_lstar_learner import MMLStarLearner as MooreMachineLearner
from pymodelextractor.teachers.moore_machines_teacher import MooreMachineTeacher as MMTeacher
from pythautomata.automata.moore_machine_automaton import MooreMachineAutomaton
from pythautomata.automata_definitions.sample_moore_machines import SampleMooreMachines
from pythautomata.model_comparators.moore_machine_comparison_strategy import MooreMachineComparisonStrategy as ComparisonStrategy
from pythautomata.automata_definitions.tomitas_grammars import TomitasGrammars as Tomitas
from pythautomata.utilities.automata_converter import AutomataConverter

class TestMMLStarLearner(unittest.TestCase):
    def setUp(self):
        self.learner = MooreMachineLearner()

    def teacher(self, automaton: MooreMachineAutomaton) -> MMTeacher:
        return MMTeacher(automaton, ComparisonStrategy())

    def test_tomitas_1(self):
        grammar1 = SampleMooreMachines.get_tomitas_automaton_1()
        teacher = self.teacher(grammar1)
        result = self.learner.learn(teacher, verbose=True)
        assert ComparisonStrategy().are_equivalent(
            result.model, grammar1)

    def test_tomitas_2(self):
        grammar2 = SampleMooreMachines.get_tomitas_automaton_2()
        teacher = self.teacher(grammar2)
        result = self.learner.learn(teacher, True)
        assert ComparisonStrategy().are_equivalent(
            result.model, grammar2)

    def test_tomitas_3(self):
        dfa = Tomitas.get_automaton_3()
        moore = AutomataConverter.convert_dfa_to_moore_machine(dfa)
        teacher = self.teacher(moore)
        result = self.learner.learn(teacher, verbose=True)
        assert ComparisonStrategy().are_equivalent(
            result.model, moore)

    def test_tomitas_4(self):
        dfa = Tomitas.get_automaton_4()
        moore = AutomataConverter.convert_dfa_to_moore_machine(dfa)
        teacher = self.teacher(moore)
        result = self.learner.learn(teacher, verbose=True)
        assert ComparisonStrategy().are_equivalent(
            result.model, moore)

    def test_tomitas_5(self):
        dfa = Tomitas.get_automaton_5()
        moore = AutomataConverter.convert_dfa_to_moore_machine(dfa)
        teacher = self.teacher(moore)
        result = self.learner.learn(teacher, verbose=True)
        assert ComparisonStrategy().are_equivalent(
            result.model, moore)
    
    def test_tomitas_6(self):
        dfa = Tomitas.get_automaton_6()
        moore = AutomataConverter.convert_dfa_to_moore_machine(dfa)
        teacher = self.teacher(moore)
        result = self.learner.learn(teacher, verbose=True)
        assert ComparisonStrategy().are_equivalent(
            result.model, moore)

    def test_tomitas_7(self):
        dfa = Tomitas.get_automaton_7()
        moore = AutomataConverter.convert_dfa_to_moore_machine(dfa)
        teacher = self.teacher(moore)
        result = self.learner.learn(teacher, verbose=True)
        assert ComparisonStrategy().are_equivalent(
            result.model, moore)