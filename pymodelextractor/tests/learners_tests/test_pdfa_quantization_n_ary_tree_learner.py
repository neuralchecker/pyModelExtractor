import unittest
from pythautomata.automata.wheighted_automaton_definition.probabilistic_deterministic_finite_automaton \
     import ProbabilisticDeterministicFiniteAutomaton
from pythautomata.automata.wheighted_automaton_definition.weighted_state import WeightedState

from pythautomata.automata_definitions.weighted_tomitas_grammars import WeightedTomitasGrammars

from pymodelextractor.learners.observation_tree_learners.pdfa_quantization_n_ary_tree_learner \
     import PDFAQuantizationNAryTreeLearner
from pymodelextractor.teachers.pac_probabilistic_teacher import PACProbabilisticTeacher
from pythautomata.model_comparators.wfa_tolerance_comparison_strategy import WFAToleranceComparator
from pythautomata.model_comparators.wfa_quantization_comparison_strategy import WFAQuantizationComparator
from pythautomata.model_comparators.wfa_partition_comparison_strategy import WFAPartitionComparator

from pymodelextractor.teachers.pdfa_teacher import PDFATeacher

from pythautomata.utilities import pdfa_generator
from pythautomata.utilities import nicaud_dfa_generator

from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.symbol import SymbolStr
from pythautomata.utilities.probability_partitioner import QuantizationProbabilityPartitioner, TopKProbabilityPartitioner
from pythautomata.utilities.guiding_wfa_sequence_generator import GuidingWDFASequenceGenerator
from pythautomata.utilities.pdfa_operations import check_is_minimal

binaryAlphabet = Alphabet(frozenset((SymbolStr('0'), SymbolStr('1'))))


class TestPDFAQuantizantionNAryTreeLearner(unittest.TestCase):

    def setUp(self):
        self.partitions = 10
        self.partitioner = QuantizationProbabilityPartitioner(self.partitions)
        self.comparator = WFAPartitionComparator(self.partitioner)
        self.learner = PDFAQuantizationNAryTreeLearner(self.partitioner)

    def test_tomitas_1(self):
        model = WeightedTomitasGrammars.get_automaton_1()
        
        teacher = PDFATeacher(model, self.comparator)
        result = self.learner.learn(teacher)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count'] > 0)        
        self.assertTrue(result.info['equivalence_queries_count'] > 0)

    def test_tomitas_2(self):
        model = WeightedTomitasGrammars.get_automaton_2()
        
        teacher = PDFATeacher(model, self.comparator)
        result = self.learner.learn(teacher)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count'] > 0)        
        self.assertTrue(result.info['equivalence_queries_count'] > 0)

    def test_tomitas_3(self):
        model = WeightedTomitasGrammars.get_automaton_3()
        
        teacher = PDFATeacher(model, self.comparator)
        result = self.learner.learn(teacher)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count'] > 0)        
        self.assertTrue(result.info['equivalence_queries_count'] > 0)

    def test_tomitas_4(self):
        model = WeightedTomitasGrammars.get_automaton_4()
        
        teacher = PDFATeacher(model, self.comparator)
        result = self.learner.learn(teacher)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count'] > 0)        
        self.assertTrue(result.info['equivalence_queries_count'] > 0)

    def test_tomitas_5(self):
        model = WeightedTomitasGrammars.get_automaton_5()
        
        teacher = PDFATeacher(model, self.comparator)
        result = self.learner.learn(teacher)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count'] > 0)        
        self.assertTrue(result.info['equivalence_queries_count'] > 0)

    def test_tomitas_6(self):
        model = WeightedTomitasGrammars.get_automaton_6()
        
        teacher = PDFATeacher(model, self.comparator)
        result = self.learner.learn(teacher)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count'] > 0)        
        self.assertTrue(result.info['equivalence_queries_count'] > 0)

    def test_tomitas_7(self):
        model = WeightedTomitasGrammars.get_automaton_7()
        
        teacher = PDFATeacher(model, self.comparator)
        result = self.learner.learn(teacher)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count'] > 0)        
        self.assertTrue(result.info['equivalence_queries_count'] > 0)

    def generate_ad_hoc_PDFA1(self):
        qeps = WeightedState("qeps", 1, 0.1)
        q0 = WeightedState("q0", 0, 0.1)
        q1 = WeightedState("q1", 0, 0.2)
        zero = SymbolStr('0')
        one = SymbolStr('1')
        qeps.add_transition(zero, q0, 0.2)
        qeps.add_transition(one, q1, 0.7)
        q0.add_transition(zero, q0, 0.5)
        q0.add_transition(one, q0, 0.4)
        q1.add_transition(zero, q1, 0.4)
        q1.add_transition(one, q1, 0.4)

        states = {qeps, q0, q1}
        comparator = WFAToleranceComparator()
        return ProbabilisticDeterministicFiniteAutomaton(binaryAlphabet, states, SymbolStr("$"), comparator,
                                                         "ad_hoc_PDFA1")

    def generate_ad_hoc_PDFA2(self):
        
        q0 = WeightedState("q0", 1, 0.8)
        q1 = WeightedState("q1", 0, 0.1)
        q2 = WeightedState("q2", 0, 0.4)
        q3 = WeightedState("q3", 0, 0.4)

        zero = SymbolStr('0')
        one = SymbolStr('1')
        q0.add_transition(zero, q2, 0.1)
        q0.add_transition(one, q1, 0.1)
        q1.add_transition(zero, q0, 0.8)
        q1.add_transition(one, q3, 0.1)
        q2.add_transition(zero, q0, 0.3)
        q2.add_transition(one, q2, 0.3)
        q3.add_transition(zero, q1, 0.3)
        q3.add_transition(one, q3, 0.3)

        states = {q0, q1, q2, q3}
        comparator = WFAToleranceComparator()
        return ProbabilisticDeterministicFiniteAutomaton(binaryAlphabet, states, SymbolStr("$"), comparator,
                                                         "ad_hoc_PDFA2")

    def generate_ad_hoc_PDFA3(self):
        
        q0 = WeightedState("q0", 1, 0.8)
        q1 = WeightedState("q1", 0, 0.1)
        q2 = WeightedState("q2", 0, 0.4)
        q3 = WeightedState("q3", 0, 0.4)
        q4 = WeightedState("q3", 0, 0.1)

        zero = SymbolStr('0')
        one = SymbolStr('1')
        q0.add_transition(zero, q0, 0.1)
        q0.add_transition(one, q1, 0.1)
        q1.add_transition(zero, q3, 0.8)
        q1.add_transition(one, q2, 0.1)
        q2.add_transition(zero, q4, 0.3)
        q2.add_transition(one, q1, 0.3)
        q3.add_transition(zero, q1, 0.3)
        q3.add_transition(one, q3, 0.3)
        q4.add_transition(zero, q4, 0.1)
        q4.add_transition(one, q1, 0.8)

        states = {q0, q1, q2, q3, q4}
        comparator = WFAToleranceComparator()
        return ProbabilisticDeterministicFiniteAutomaton(binaryAlphabet, states, SymbolStr("$"), comparator,
                                                         "ad_hoc_PDFA3")

    def generate_ad_hoc_PDFA4(self):
        
        qeps = WeightedState("qeps", 1, 0.8)
        q0 = WeightedState("q0", 0, 0.4)
        q1 = WeightedState("q1", 0, 0.4)
        q11 = WeightedState("q11", 0, 0.1)

        zero = SymbolStr('0')
        one = SymbolStr('1')
        qeps.add_transition(zero, q0, 0.1)
        qeps.add_transition(one, q1, 0.1)
        q0.add_transition(zero, qeps, 0.3)
        q0.add_transition(one, q1, 0.3)
        q1.add_transition(zero, q0, 0.3)
        q1.add_transition(one, q11, 0.3)        
        q11.add_transition(zero, q0, 0.1)
        q11.add_transition(one, q11, 0.8)

        states = {qeps, q0, q1, q11}
        comparator = WFAToleranceComparator()
        return ProbabilisticDeterministicFiniteAutomaton(binaryAlphabet, states, SymbolStr("$"), comparator,
                                                         "ad_hoc_PDFA4")

    def generate_ad_hoc_PDFA5(self):
        
        qeps = WeightedState("qeps", 1, 0.4223)
        q0 = WeightedState("q0", 0, 0.36321)
        q00 = WeightedState("q1", 0, 0.62862)
        q1 = WeightedState("q11", 0, 0.50343)

        zero = SymbolStr('0')
        one = SymbolStr('1')
        qeps.add_transition(zero, q0, 0.34426)
        qeps.add_transition(one, q1, 0.23344)
        q0.add_transition(zero, q00, 0.35478)
        q0.add_transition(one, q0, 0.28201)
        q00.add_transition(zero, q00, 0.16799)
        q00.add_transition(one, q1, 0.20339)        
        q1.add_transition(zero, q00, 0.3603)
        q1.add_transition(one, q00, 0.13627)

        states = {qeps, q0, q00, q1}
        comparator = WFAToleranceComparator()
        return ProbabilisticDeterministicFiniteAutomaton(binaryAlphabet, states, SymbolStr("$"), comparator,
                                                         "ad_hoc_PDFA5")
    
    def generate_ad_hoc_PDFA6(self):
        
        qeps = WeightedState("qeps", 1, 0.60296)
        q1 = WeightedState("q0", 0, 0.56036)
        q10 = WeightedState("q1", 0, 0.50998)
        q0 = WeightedState("q11", 0, 0.26139)

        zero = SymbolStr('0')
        one = SymbolStr('1')
        qeps.add_transition(zero, q0, 0.12317)
        qeps.add_transition(one, q1, 0.27387)
        q1.add_transition(zero, q10, 0.20647)
        q1.add_transition(one, q1, 0.23317)
        q10.add_transition(zero, q10, 0.25791)
        q10.add_transition(one, q0, 0.23211)        
        q0.add_transition(zero, q1, 0.19049)
        q0.add_transition(one, q0, 0.54812)

        states = {qeps, q0, q10, q1}
        comparator = WFAToleranceComparator()
        return ProbabilisticDeterministicFiniteAutomaton(binaryAlphabet, states, SymbolStr("$"), comparator,
                                                         "ad_hoc_PDFA6")

    def test_ad_hoc_PDFA1(self):
        model = self.generate_ad_hoc_PDFA1()
        
        teacher = PDFATeacher(model, self.comparator)
        result = self.learner.learn(teacher)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count'] > 0)        
        self.assertTrue(result.info['equivalence_queries_count'] > 0)

    def test_ad_hoc_PDFA2(self):   
        model = self.generate_ad_hoc_PDFA2()
        
        teacher = PDFATeacher(model, self.comparator)
        result = self.learner.learn(teacher)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count'] > 0)        
        self.assertTrue(result.info['equivalence_queries_count'] > 0)

    def test_ad_hoc_PDFA3(self):     
        model = self.generate_ad_hoc_PDFA3()
        
        teacher = PDFATeacher(model, self.comparator)
        result = self.learner.learn(teacher)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count'] > 0)        
        self.assertTrue(result.info['equivalence_queries_count'] > 0)

    def test_ad_hoc_PDFA4(self):     
        model = self.generate_ad_hoc_PDFA4()
        
        teacher = PDFATeacher(model, self.comparator)
        result = self.learner.learn(teacher)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count'] > 0)        
        self.assertTrue(result.info['equivalence_queries_count'] > 0)

    def test_ad_hoc_PDFA5(self):     
        model = self.generate_ad_hoc_PDFA5()
        
        teacher = PDFATeacher(model, self.comparator)
        result = self.learner.learn(teacher)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count'] > 0)        
        self.assertTrue(result.info['equivalence_queries_count'] > 0)
    
    def test_ad_hoc_PDFA6(self):     
        model = self.generate_ad_hoc_PDFA6()
        
        teacher = PDFATeacher(model, self.comparator)
        result = self.learner.learn(teacher)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count'] > 0)        
        self.assertTrue(result.info['equivalence_queries_count'] > 0)

    def generate_random_pdfas(self, sizes, n):
        pdfas = []        
        for size in sizes:
            for i in range(n):
                dfa = nicaud_dfa_generator.generate_dfa(alphabet = binaryAlphabet, nominal_size=size, seed=i)
                dfa.name = "random_DFA_nominal_size_"+str(size)+"_"+str(i)
                pdfa = pdfa_generator.pdfa_from_dfa(dfa)
                pdfa.name = "random_PDFA_nominal_size_"+str(size)+"_"+str(i)
                pdfas.append(pdfa)
        return pdfas


    def test_against_random_PDFAs(self):
        models = self.generate_random_pdfas(sizes=[7, 10, 20], n=20)
        for model in models:
            #print('Extracting model:', model.name)
            partitions = 20
            other_comparator = WFAQuantizationComparator(partitions)
            partitioner = QuantizationProbabilityPartitioner(partitions)
            teacher = PDFATeacher(model, other_comparator)
            other_learner = PDFAQuantizationNAryTreeLearner(partitioner)
            result = other_learner.learn(teacher)
            extracted_model = result.model           
            self.assertTrue(other_comparator.get_counterexample_between(model, extracted_model) is None)
            self.assertTrue(result.info['last_token_weight_queries_count'] > 0)        
            self.assertTrue(result.info['equivalence_queries_count'] > 0)

    def generate_PDFA_with_3_equivalent_states_according_to_argmax(self):
        
        qeps = WeightedState("qeps", 1, 0.60296)
        q1 = WeightedState("q0", 0, 0.56036)
        q11 = WeightedState("q1", 0, 0.50998)

        zero = SymbolStr('0')
        one = SymbolStr('1')
        qeps.add_transition(zero, qeps, 0.12317)
        qeps.add_transition(one, q1, 0.27387)
        q1.add_transition(zero, q1, 0.20647)
        q1.add_transition(one, q11, 0.23317)
        q11.add_transition(zero, q11, 0.25791)
        q11.add_transition(one, q11, 0.23211)        
        

        states = {qeps, q1, q11}
        comparator = WFAToleranceComparator()
        return ProbabilisticDeterministicFiniteAutomaton(binaryAlphabet, states, SymbolStr("$"), comparator,
                                                         "PDFA_with_3_equivalent_states_according_to_argmax")
    
    def generate_PDFA_with_3_different_states_according_to_argmax(self):
        
        qeps = WeightedState("qeps", 1, 0.6)
        q1 = WeightedState("q0", 0, 0.2)
        q11 = WeightedState("q1", 0, 0.2)

        zero = SymbolStr('0')
        one = SymbolStr('1')
        qeps.add_transition(zero, qeps, 0.2)
        qeps.add_transition(one, q1, 0.2)
        q1.add_transition(zero, q1, 0.6)
        q1.add_transition(one, q11, 0.2)
        q11.add_transition(zero, q11, 0.2)
        q11.add_transition(one, q11, 0.6)      
        states = {qeps, q1, q11}
        comparator = WFAToleranceComparator()
        return ProbabilisticDeterministicFiniteAutomaton(binaryAlphabet, states, SymbolStr("$"), comparator,
                                                         "PDFA_with_3_equivalent_states_according_to_argmax")
    

    def test_arg_max_partitioner1(self):     
        model = self.generate_PDFA_with_3_equivalent_states_according_to_argmax()
        partitioner = TopKProbabilityPartitioner(1)
        comparator = WFAPartitionComparator(partitioner)
        teacher = PDFATeacher(model, comparator)
        learner = PDFAQuantizationNAryTreeLearner(partitioner)
        result = learner.learn(teacher)
        extracted_model = result.model
        self.assertTrue(len(extracted_model.weighted_states) == 1)
        self.assertTrue(result.info['last_token_weight_queries_count'] > 0)        
        self.assertTrue(result.info['equivalence_queries_count'] > 0)

    
    def test_arg_max_partitioner2(self):     
        model = self.generate_PDFA_with_3_different_states_according_to_argmax()
        partitioner = TopKProbabilityPartitioner(1)
        comparator = WFAPartitionComparator(partitioner)
        teacher = PDFATeacher(model, comparator)
        learner = PDFAQuantizationNAryTreeLearner(partitioner)
        result = learner.learn(teacher)
        extracted_model = result.model
        self.assertTrue(len(extracted_model.weighted_states) == 3)
        self.assertTrue(result.info['last_token_weight_queries_count'] > 0)        
        self.assertTrue(result.info['equivalence_queries_count'] > 0)

    def generate_PDFA_with_section_reachable_through_zero_transitions(self):        
        qeps = WeightedState("qeps", 1, 0.6)
        q1 = WeightedState("q0", 0, 0.5)
        q11 = WeightedState("q1", 0, 0.2)

        zero = SymbolStr('0')
        one = SymbolStr('1')
        qeps.add_transition(zero, qeps, 0.4)
        qeps.add_transition(one, q1, 0)
        q1.add_transition(zero, q1, 0)
        q1.add_transition(one, q11, 0.5)
        q11.add_transition(zero, q11, 0)
        q11.add_transition(one, q1, 0.8)      
        states = {qeps, q1, q11}
        comparator = WFAQuantizationComparator(100)
        return ProbabilisticDeterministicFiniteAutomaton(binaryAlphabet, states, SymbolStr("$"), comparator,
                                                         "PDFA_with_section_reachable_through_zero_transitions")
    
    
    def generate_PDFA_with_section_reachable_through_zero_transitions2(self):        
        qeps = WeightedState("qeps", 1, 0.4)
        q1 = WeightedState("q0", 0, 0.5)
        q11 = WeightedState("q1", 0, 0.2)

        zero = SymbolStr('0')
        one = SymbolStr('1')
        qeps.add_transition(zero, qeps, 0.4)
        qeps.add_transition(one, q1, 0.2)
        q1.add_transition(zero, q1, 0.5)
        q1.add_transition(one, q11, 0)
        q11.add_transition(zero, q11, 0)
        q11.add_transition(one, q1, 0.8)      
        states = {qeps, q1, q11}
        comparator = WFAQuantizationComparator(100)
        return ProbabilisticDeterministicFiniteAutomaton(binaryAlphabet, states, SymbolStr("$"), comparator,
                                                         "PDFA_with_section_reachable_through_zero_transitions")
    
    def test_learn_exact_PDFA_w_zero_transition(self):     
        model = self.generate_PDFA_with_section_reachable_through_zero_transitions()
        partitioner = QuantizationProbabilityPartitioner(100)
        comparator = WFAPartitionComparator(partitioner)        
        teacher = PDFATeacher(model, comparator)
        learner = PDFAQuantizationNAryTreeLearner(partitioner, omit_zero_transitions=False)
        result = learner.learn(teacher)
        extracted_model = result.model
        self.assertTrue(len(extracted_model.weighted_states) == 3)
    
    def test_learn_approximate_PDFA_w_zero_transition(self):     
        model = self.generate_PDFA_with_section_reachable_through_zero_transitions()
        partitioner = QuantizationProbabilityPartitioner(100)
        comparator = WFAPartitionComparator(partitioner)
        generator = GuidingWDFASequenceGenerator(model,5)
        teacher = PACProbabilisticTeacher(model, comparator, sequence_generator=generator)
        learner = PDFAQuantizationNAryTreeLearner(partitioner, omit_zero_transitions=True, check_probabilistic_hipothesis=False)
        result = learner.learn(teacher)
        extracted_model = result.model
        self.assertTrue(len(extracted_model.weighted_states) == 2)
        # This test checks that the model is minimal. But check_is_minimal function also goes through the states and check that the
        # transitions are correct. Which is something that was broken in the past. 
        self.assertTrue(check_is_minimal(extracted_model))

    def test_learn_approximate_PDFA_w_zero_transition2(self):     
        model = self.generate_PDFA_with_section_reachable_through_zero_transitions2()
        partitioner = QuantizationProbabilityPartitioner(100)
        comparator = WFAPartitionComparator(partitioner)
        generator = GuidingWDFASequenceGenerator(model,500, random_stop_proba=0.001)
        teacher = PACProbabilisticTeacher(model, comparator, sequence_generator=generator)
        learner = PDFAQuantizationNAryTreeLearner(partitioner, omit_zero_transitions=True, check_probabilistic_hipothesis=False)
        result = learner.learn(teacher)
        extracted_model = result.model
        self.assertTrue(len(extracted_model.weighted_states) == 3) # 2 states + hole
        # This test checks that the model is minimal. But this function also goes through the states and check that the
        # transitions are correct. Which is something that was broken in the past. 
        self.assertTrue(check_is_minimal(extracted_model))