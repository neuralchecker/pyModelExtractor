import unittest

from pymodelextractor.learners.other_learners.ensemble_probabilistic_learner import \
    EnsembleProbabilisticLearner
from pymodelextractor.learners.observation_tree_learners.bounded_pdfa_quantization_n_ary_tree_learner import BoundedPDFAQuantizationNAryTreeLearner
from pymodelextractor.teachers.pac_probabilistic_teacher import \
    PACProbabilisticTeacher
from pythautomata.automata.wheighted_automaton_definition.probabilistic_deterministic_finite_automaton import \
    ProbabilisticDeterministicFiniteAutomaton
from pythautomata.automata_definitions.weighted_tomitas_grammars import WeightedTomitasGrammars
from pythautomata.utilities.uniform_word_sequence_generator import UniformWordSequenceGenerator
from pythautomata.model_comparators.wfa_partition_comparison_strategy import WFAPartitionComparator
from pythautomata.utilities.probability_partitioner import QuantizationProbabilityPartitioner
from itertools import chain
from pythautomata.base_types.sequence import Sequence


class TestEnsembleProbabilisticLearner(unittest.TestCase):
    def setUp(self):
        self.partitioner = QuantizationProbabilityPartitioner(10)
        self.comparator = WFAPartitionComparator(self.partitioner)
        max_states = max_query_length = 1000
        l1 = BoundedPDFAQuantizationNAryTreeLearner(self.partitioner, max_states, max_query_length).learn
        l2 = BoundedPDFAQuantizationNAryTreeLearner(self.partitioner, max_states, max_query_length).learn
        l3 = BoundedPDFAQuantizationNAryTreeLearner(self.partitioner, max_states, max_query_length).learn
        self.learner = EnsembleProbabilisticLearner(learning_functions=[l1, l2, l3])

    def teacher(self, automaton: ProbabilisticDeterministicFiniteAutomaton) -> PACProbabilisticTeacher:
        epsilon = delta = 0.05
        return PACProbabilisticTeacher(automaton, self.comparator,epsilon, delta, UniformWordSequenceGenerator(automaton.alphabet, 10))
    
    def evaluate_results(self, target, learned):
        generator = UniformWordSequenceGenerator(target.alphabet, 10, random_seed=10)
        data_test = list(generator.generate_words(100))
        suffixes = list()
        suffixes.append(Sequence() + learned.terminal_symbol)
        for symbol in learned.alphabet.symbols:
            suffixes.append(Sequence((symbol,)))

        oracle_results = [target.last_token_probabilities(word, suffixes) for word in data_test]
        learned_results = [learned.last_token_probabilities(word, suffixes) for word in data_test]     
        for i in range(len(oracle_results)):
            self.assertTrue(self.partitioner.are_in_same_partition(oracle_results[i], learned_results[i]))

    def test_tomitas_1(self):
        grammar1 = WeightedTomitasGrammars.get_automaton_1()
        teacher = self.teacher(grammar1)
        model = self.learner.learn(teacher).model
        self.evaluate_results(grammar1, model)

    def test_tomitas_2(self):
        grammar2 = WeightedTomitasGrammars.get_automaton_2()
        teacher = self.teacher(grammar2)
        model = self.learner.learn(teacher).model
        self.evaluate_results(grammar2, model)

    def test_tomitas_3(self):
        grammar3 = WeightedTomitasGrammars.get_automaton_3()
        teacher = self.teacher(grammar3)
        model = self.learner.learn(teacher).model
        self.evaluate_results(grammar3, model)

    def test_tomitas_4(self):
        grammar4 = WeightedTomitasGrammars.get_automaton_4()
        teacher = self.teacher(grammar4)
        model = self.learner.learn(teacher).model
        self.evaluate_results(grammar4, model)

    def test_against_many_DFAs(self):
        all_automata = WeightedTomitasGrammars.get_all_automata()                                    
        for automaton in all_automata:
            teacher = self.teacher(automaton)
            model = self.learner.learn(teacher).model
            self.evaluate_results(automaton, model)
        