import unittest

from pymodelextractor.learners.other_learners.ensemble_boolean_learner import \
    EnsembleBooleanLearner
from pymodelextractor.learners.observation_table_learners.lstar_learner import LStarLearner
from pymodelextractor.teachers.pac_boolean_teacher import \
    PACBooleanTeacher
from pythautomata.automata.deterministic_finite_automaton import \
    DeterministicFiniteAutomaton
from pythautomata.automata_definitions.tomitas_grammars import TomitasGrammars
from pythautomata.automata_definitions.bollig_habermehl_kern_leucker_automata import BolligHabermehlKernLeuckerAutomata
from pythautomata.automata_definitions.omlin_giles_automata import OmlinGilesAutomata
from pythautomata.model_comparators.hopcroft_karp_comparison_strategy import \
    HopcroftKarpComparisonStrategy as ComparisonStrategy
from pythautomata.utilities.uniform_word_sequence_generator import UniformWordSequenceGenerator
from itertools import chain



class TestEnsembleBooleanLearner(unittest.TestCase):
    def setUp(self):
        l1 = LStarLearner().learn
        l2 = LStarLearner().learn
        l3 = LStarLearner().learn
        self.learner = EnsembleBooleanLearner(learning_functions=[l1, l2, l3])

    def teacher(self, automaton: DeterministicFiniteAutomaton) -> PACBooleanTeacher:
        epsilon = delta = 0.05
        return PACBooleanTeacher(automaton, epsilon, delta, UniformWordSequenceGenerator(automaton.alphabet, 10))
    
    def evaluate_results(self, target, learned):
        generator = UniformWordSequenceGenerator(target.alphabet, 10, random_seed=10)
        data_test = list(generator.generate_words(100))

        oracle_results = [target.accepts(word) for word in data_test]
        learned_results = [learned.accepts(word) for word in data_test]        
        self.assertTrue(oracle_results == learned_results)

    def test_tomitas_1(self):
        grammar1 = TomitasGrammars.get_automaton_1()
        teacher = self.teacher(grammar1)
        model = self.learner.learn(teacher).model
        self.evaluate_results(grammar1, model)

    def test_tomitas_2(self):
        grammar2 = TomitasGrammars.get_automaton_2()
        teacher = self.teacher(grammar2)
        model = self.learner.learn(teacher).model
        self.evaluate_results(grammar2, model)

    def test_tomitas_3(self):
        grammar3 = TomitasGrammars.get_automaton_3()
        teacher = self.teacher(grammar3)
        model = self.learner.learn(teacher).model
        self.evaluate_results(grammar3, model)

    def test_tomitas_4(self):
        grammar4 = TomitasGrammars.get_automaton_4()
        teacher = self.teacher(grammar4)
        model = self.learner.learn(teacher).model
        self.evaluate_results(grammar4, model)

    def test_against_many_DFAs(self):
        mergedAutomata = list(chain(TomitasGrammars.get_all_automata(),
                                    BolligHabermehlKernLeuckerAutomata.get_all_automata(),
                                    OmlinGilesAutomata.get_all_automata()))
        for automaton in mergedAutomata:
            teacher = self.teacher(automaton)
            model = self.learner.learn(teacher).model
            self.evaluate_results(automaton, model)
        