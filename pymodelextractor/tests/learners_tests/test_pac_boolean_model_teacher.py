from re import I
import unittest
from numpy import result_type
from pythautomata.abstract.boolean_model import BooleanModel
from pythautomata.automata.deterministic_finite_automaton import DeterministicFiniteAutomaton


from pythautomata.automata_definitions.bollig_habermehl_kern_leucker_automata import BolligHabermehlKernLeuckerAutomata
from pythautomata.automata_definitions.omlin_giles_automata import OmlinGilesAutomata
from pythautomata.automata_definitions.other_automata import OtherAutomata
from pythautomata.automata_definitions.tomitas_grammars import TomitasGrammars
from pythautomata.model_comparators.hopcroft_karp_comparison_strategy import \
    HopcroftKarpComparisonStrategy as ComparisonStrategy

from pymodelextractor.learners.observation_table_learners.lstar_learner import LStarLearner
from pymodelextractor.learners.observation_tree_learners.kearns_vazirani_learner import KearnsVaziraniLearner

from itertools import chain

from pymodelextractor.teachers.pac_boolean_teacher import PACBooleanTeacher
from pythautomata.utilities.regex_generator import RegularExpressionGenerator

class TestPACBooleanModelTeachers(unittest.TestCase):

    def setUp(self):
        self.learners = [LStarLearner()]
    
    def teacher(self, model: BooleanModel) -> PACBooleanTeacher:
        return PACBooleanTeacher(model, 0.005, 0.005,max_seq_length=20)

    def test_against_many_DFAs(self):
        mergedAutomata = list(chain(TomitasGrammars.get_all_automata()))
        for automaton in mergedAutomata:
            for learner in self.learners:
                teacher = self.teacher(automaton)
                result = learner.learn(teacher)
                assert ComparisonStrategy().are_equivalent(result.model, automaton)

    def test_against_random_regex(self):
        alphabet = TomitasGrammars.get_automaton_1().alphabet
        for i in range(20):
            for learner in self.learners:
                rand_regex = RegularExpressionGenerator(alphabet).generate_regular_expression_with(iterations=20, some_seed = i)
                teacher = self.teacher(rand_regex)
                result = learner.learn(teacher)                
                assert True
