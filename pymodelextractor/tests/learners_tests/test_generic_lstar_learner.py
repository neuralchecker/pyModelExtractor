
import unittest

from pymodelextractor.teachers.generic_teacher import \
    GenericTeacher
from pymodelextractor.learners.observation_table_learners.lstar_factory import LStarFactory
from pythautomata.automata_definitions.tomitas_grammars import TomitasGrammars
from pythautomata.model_comparators.hopcroft_karp_comparison_strategy import \
    HopcroftKarpComparisonStrategy as ComparisonStrategy
from pythautomata.model_comparators.moore_machine_comparison_strategy import MooreMachineComparisonStrategy
from pythautomata.model_comparators.dfa_comparison_strategy import DFAComparisonStrategy
from pythautomata.utilities.automata_converter import AutomataConverter
from pythautomata.utilities.simple_dfa_generator import generate_dfa
from pymodelextractor.utils.time_bound_utilities import is_unix_system


class TestGenericLStarLearner(unittest.TestCase):
    def test_generic_learner(self):
        alphabet = TomitasGrammars.get_automaton_7()._alphabet
        iters = 10
        for i in range(2):
            num = (i+1)*100
            for _ in range(iters):
                automaton = generate_dfa(alphabet, num)
                moore = AutomataConverter().convert_dfa_to_moore_machine(automaton)

                result = LStarFactory.get_dfa_lstar_learner().learn(GenericTeacher(automaton, DFAComparisonStrategy()))
                assert ComparisonStrategy().are_equivalent(result.model, automaton)

                result = LStarFactory.get_moore_machine_lstar_learner().learn(GenericTeacher(moore, MooreMachineComparisonStrategy()))
                assert MooreMachineComparisonStrategy().are_equivalent(result.model, moore)

    def test_bounded_generic_learner_tomitas1(self):
        automaton = TomitasGrammars.get_automaton_1()
        moore = AutomataConverter().convert_dfa_to_moore_machine(automaton)

        result = LStarFactory.get_dfa_lstar_learner(max_query_length=10, max_states=10).learn(GenericTeacher(automaton, DFAComparisonStrategy()))
        assert ComparisonStrategy().are_equivalent(result.model, automaton)

        result = LStarFactory.get_moore_machine_lstar_learner(max_query_length=10, max_states=10).learn(GenericTeacher(moore, MooreMachineComparisonStrategy()))
        assert MooreMachineComparisonStrategy().are_equivalent(result.model, moore)

    def test_bounded_generic_learner_tomitas2(self):
        automaton = TomitasGrammars.get_automaton_2()
        moore = AutomataConverter().convert_dfa_to_moore_machine(automaton)

        result = LStarFactory.get_dfa_lstar_learner(max_query_length=10, max_states=10).learn(GenericTeacher(automaton, DFAComparisonStrategy()))
        assert ComparisonStrategy().are_equivalent(result.model, automaton)

        result = LStarFactory.get_moore_machine_lstar_learner(max_query_length=10, max_states=10).learn(GenericTeacher(moore, MooreMachineComparisonStrategy()))
        assert MooreMachineComparisonStrategy().are_equivalent(result.model, moore)

    def test_bounded_generic_learner_tomitas3(self):
        automaton = TomitasGrammars.get_automaton_3()
        moore = AutomataConverter().convert_dfa_to_moore_machine(automaton)

        result = LStarFactory.get_dfa_lstar_learner(max_query_length=10, max_states=10).learn(GenericTeacher(automaton, DFAComparisonStrategy()))
        assert ComparisonStrategy().are_equivalent(result.model, automaton)

        result = LStarFactory.get_moore_machine_lstar_learner(max_query_length=10, max_states=10).learn(GenericTeacher(moore, MooreMachineComparisonStrategy()))
        assert MooreMachineComparisonStrategy().are_equivalent(result.model, moore)

    def test_bounded_generic_learner_tomitas4(self):
        automaton = TomitasGrammars.get_automaton_4()
        moore = AutomataConverter().convert_dfa_to_moore_machine(automaton)

        result = LStarFactory.get_dfa_lstar_learner(max_query_length=10, max_states=10).learn(GenericTeacher(automaton, DFAComparisonStrategy()))
        assert ComparisonStrategy().are_equivalent(result.model, automaton)

        result = LStarFactory.get_moore_machine_lstar_learner(max_query_length=10, max_states=10).learn(GenericTeacher(moore, MooreMachineComparisonStrategy()))
        assert MooreMachineComparisonStrategy().are_equivalent(result.model, moore)

    def test_bounded_generic_learner_tomitas5(self):
        automaton = TomitasGrammars.get_automaton_5()
        moore = AutomataConverter().convert_dfa_to_moore_machine(automaton)

        result = LStarFactory.get_dfa_lstar_learner(max_query_length=10, max_states=10).learn(GenericTeacher(automaton, DFAComparisonStrategy()))
        assert ComparisonStrategy().are_equivalent(result.model, automaton)

        result = LStarFactory.get_moore_machine_lstar_learner(max_query_length=10, max_states=10).learn(GenericTeacher(moore, MooreMachineComparisonStrategy()))
        assert MooreMachineComparisonStrategy().are_equivalent(result.model, moore)

    def test_bounded_generic_learner_big_automaton(self):
        automaton = generate_dfa(TomitasGrammars.get_automaton_7()._alphabet, 100)
        moore = AutomataConverter().convert_dfa_to_moore_machine(automaton)

        result = LStarFactory.get_dfa_lstar_learner(max_states=10).learn(GenericTeacher(automaton, DFAComparisonStrategy()))
        self.assertFalse(ComparisonStrategy().are_equivalent(result.model, automaton))

        result = LStarFactory.get_moore_machine_lstar_learner(max_states=10).learn(GenericTeacher(moore, MooreMachineComparisonStrategy()))
        self.assertFalse(MooreMachineComparisonStrategy().are_equivalent(result.model, moore))

        result = LStarFactory.get_dfa_lstar_learner(max_query_length=10).learn(GenericTeacher(automaton, DFAComparisonStrategy()))
        self.assertFalse(ComparisonStrategy().are_equivalent(result.model, automaton))

        result = LStarFactory.get_moore_machine_lstar_learner(max_query_length=10).learn(GenericTeacher(moore, MooreMachineComparisonStrategy()))
        self.assertFalse(MooreMachineComparisonStrategy().are_equivalent(result.model, moore))

    def test_time_bounded_lstar(self):

        if is_unix_system():
            automaton = generate_dfa(TomitasGrammars.get_automaton_7()._alphabet, 10000)
            moore = AutomataConverter().convert_dfa_to_moore_machine(automaton)

            
            result = LStarFactory.get_dfa_lstar_learner(max_time=1).learn(GenericTeacher(automaton, DFAComparisonStrategy()))
            self.assertFalse(ComparisonStrategy().are_equivalent(result.model, automaton))

            result = LStarFactory.get_moore_machine_lstar_learner(max_time=1).learn(GenericTeacher(moore, MooreMachineComparisonStrategy()))
            self.assertFalse(MooreMachineComparisonStrategy().are_equivalent(result.model, moore))


        
    

    