import unittest

from pymodelextractor.learners.observation_table_learners.general_lstar_learner import GeneralLStarLearner
from pymodelextractor.teachers.general_teacher import GeneralTeacher
from pythautomata.automata_definitions.sample_moore_machines import SampleMooreMachines
from pythautomata.model_comparators.moore_machine_comparison_strategy import MooreMachineComparisonStrategy as ComparisonStrategy
from pythautomata.model_comparators.random_walk_mm_comparison_strategy import RandomWalkMMComparisonStrategy
from pythautomata.automata_definitions.tomitas_grammars import TomitasGrammars as Tomitas
from pythautomata.utilities.nicaud_mm_generator import generate_moore_machine
from pythautomata.utilities.automata_converter import AutomataConverter
from pymodelextractor.learners.observation_table_learners.translators.mm_observation_table_translator import MMObservationTableTranslator

class TestMMLStarLearner(unittest.TestCase):
    def setUp(self):
        self.learner = GeneralLStarLearner(MMObservationTableTranslator())

    def teacher(self, automaton):
        return GeneralTeacher(automaton, ComparisonStrategy())

    def test_tomitas_1_without_log(self):
        grammar1 = SampleMooreMachines.get_tomitas_automaton_1()
        teacher = self.teacher(grammar1)
        result = self.learner.learn(teacher, log_hierachy=0)
        assert ComparisonStrategy().are_equivalent(
            result.model, grammar1)
        
    def test_tomitas_1_with_log_hierachy_info(self):
        grammar1 = SampleMooreMachines.get_tomitas_automaton_1()
        teacher = self.teacher(grammar1)
        result = self.learner.learn(teacher, log_hierachy=1)
        assert ComparisonStrategy().are_equivalent(
            result.model, grammar1)
        
    def test_tomitas_1_with_log_hierachy_debug(self):
        grammar1 = SampleMooreMachines.get_tomitas_automaton_1()
        teacher = self.teacher(grammar1)
        result = self.learner.learn(teacher, log_hierachy=2)
        assert ComparisonStrategy().are_equivalent(
            result.model, grammar1)
        
    def test_tomitas_1(self):
        grammar1 = SampleMooreMachines.get_tomitas_automaton_1()
        teacher = self.teacher(grammar1)
        result = self.learner.learn(teacher, log_hierachy=3)
        assert ComparisonStrategy().are_equivalent(
            result.model, grammar1)
    
    def test_tomitas_2(self):
        grammar2 = SampleMooreMachines.get_tomitas_automaton_2()
        teacher = self.teacher(grammar2)
        result = self.learner.learn(teacher)
        assert ComparisonStrategy().are_equivalent(
            result.model, grammar2)

    def test_tomitas_3(self):
        dfa = Tomitas.get_automaton_3()
        moore = AutomataConverter.convert_dfa_to_moore_machine(dfa)
        teacher = self.teacher(moore)
        result = self.learner.learn(teacher)
        assert ComparisonStrategy().are_equivalent(
            result.model, moore)

    def test_tomitas_4(self):
        dfa = Tomitas.get_automaton_4()
        moore = AutomataConverter.convert_dfa_to_moore_machine(dfa)
        teacher = self.teacher(moore)
        result = self.learner.learn(teacher)
        assert ComparisonStrategy().are_equivalent(
            result.model, moore)

    def test_tomitas_5(self):
        dfa = Tomitas.get_automaton_5()
        moore = AutomataConverter.convert_dfa_to_moore_machine(dfa)
        teacher = self.teacher(moore)
        result = self.learner.learn(teacher)
        assert ComparisonStrategy().are_equivalent(
            result.model, moore)
    
    def test_tomitas_6(self):
        dfa = Tomitas.get_automaton_6()
        moore = AutomataConverter.convert_dfa_to_moore_machine(dfa)
        teacher = self.teacher(moore)
        result = self.learner.learn(teacher)
        assert ComparisonStrategy().are_equivalent(
            result.model, moore)

    def test_tomitas_7(self):
        dfa = Tomitas.get_automaton_7()
        moore = AutomataConverter.convert_dfa_to_moore_machine(dfa)
        teacher = self.teacher(moore)
        result = self.learner.learn(teacher)
        assert ComparisonStrategy().are_equivalent(
            result.model, moore)

    def test_random_walk_lstar(self):
        mm = SampleMooreMachines.get_3_states_automaton()
        moore = generate_moore_machine(mm._alphabet, mm._output_alphabet, 300, 21)
        teacher = GeneralTeacher(moore, RandomWalkMMComparisonStrategy(10000, 0.01))
        result = self.learner.learn(teacher)
        assert ComparisonStrategy().are_equivalent(
            result.model, moore)