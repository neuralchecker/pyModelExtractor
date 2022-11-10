import unittest

from pymodelextractor.learners.observation_table_learners.mm_lstar_learner import MooreMachineLearner
from pymodelextractor.teachers.moore_machines_teacher import MooreMachineTeacher as MMTeacher
from pythautomata.automata.moore_machines_automaton import MooreMachineAutomaton
from pythautomata.automata_definitions.sample_moore_machines import SampleMooreMachines
from pythautomata.model_comparators.moore_machine_comparison_strategy import MooreMachineComparisonStrategy as ComparisonStrategy

class TestMMLStarLearner(unittest.TestCase):
    def setUp(self):
        self.learner = MooreMachineLearner()

    def teacher(self, automaton: MooreMachineAutomaton) -> MMTeacher:
        return MMTeacher(automaton, ComparisonStrategy())

    def test_tomitas_1(self):
        grammar1 = SampleMooreMachines.get_automaton_1()
        teacher = self.teacher(grammar1)
        result = self.learner.learn(teacher)
        assert ComparisonStrategy().are_equivalent(
            result.model, grammar1)

    def test_tomitas_2(self):
        grammar2 = SampleMooreMachines.get_automaton_2()
        teacher = self.teacher(grammar2)
        result = self.learner.learn(teacher)
        assert ComparisonStrategy().are_equivalent(
            result.model, grammar2)