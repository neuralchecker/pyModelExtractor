import unittest
from numpy import result_type

from pythautomata.automata_definitions.weighted_tomitas_grammars import WeightedTomitasGrammars

from pymodelextractor.learners.observation_tree_learners.pdfa_quantization_n_ary_tree_learner import PDFAQuantizationNAryTreeLearner

from pymodelextractor.teachers.pdfa_teacher import PDFATeacher
from pymodelextractor.teachers.sample_probabilistic_teacher import SampleProbabilisticTeacher
from pymodelextractor.teachers.pac_batch_probabilistic_teacher import PACBatchProbabilisticTeacher
from pythautomata.model_comparators.wfa_quantization_comparison_strategy import WFAQuantizationComparator

class TestPACBatchTeacherQuant(unittest.TestCase):

    def setUp(self):
        self.learner = PDFAQuantizationNAryTreeLearner()

    def test_tomitas_1(self):
        model = WeightedTomitasGrammars.get_automaton_1()
        partitions = 10
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = WFAQuantizationComparator(partitions), max_seq_length=20)
        result = self.learner.learn(teacher, partitions = partitions, verbose = True)        
        extracted_model = result.model     
        self.assertTrue(WFAQuantizationComparator(partitions).are_equivalent(model, extracted_model))
    
    def test_tomitas_2(self):
        model = WeightedTomitasGrammars.get_automaton_2()
        partitions = 10
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = WFAQuantizationComparator(partitions), max_seq_length=20)
        result = self.learner.learn(teacher, partitions = partitions, verbose = True)        
        extracted_model = result.model     
        self.assertTrue(WFAQuantizationComparator(partitions).are_equivalent(model, extracted_model))
    

    def test_tomitas_3(self):
        model = WeightedTomitasGrammars.get_automaton_3()
        partitions = 10
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = WFAQuantizationComparator(partitions), max_seq_length=20)
        result = self.learner.learn(teacher, partitions = partitions, verbose = True)        
        extracted_model = result.model     
        self.assertTrue(WFAQuantizationComparator(partitions).are_equivalent(model, extracted_model))

    def test_tomitas_4(self):
        model = WeightedTomitasGrammars.get_automaton_4()
        partitions = 10
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = WFAQuantizationComparator(partitions), max_seq_length=20)
        result = self.learner.learn(teacher, partitions = partitions, verbose = True)        
        extracted_model = result.model     
        self.assertTrue(WFAQuantizationComparator(partitions).are_equivalent(model, extracted_model))
    
    def test_tomitas_5(self):
        model = WeightedTomitasGrammars.get_automaton_5()
        partitions = 10
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = WFAQuantizationComparator(partitions), max_seq_length=20)
        result = self.learner.learn(teacher, partitions = partitions, verbose = True)        
        extracted_model = result.model     
        self.assertTrue(WFAQuantizationComparator(partitions).are_equivalent(model, extracted_model))
    

    def test_tomitas_6(self):
        model = WeightedTomitasGrammars.get_automaton_6()
        partitions = 10
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = WFAQuantizationComparator(partitions), max_seq_length=20)
        result = self.learner.learn(teacher, partitions = partitions, verbose = True)        
        extracted_model = result.model     
        self.assertTrue(WFAQuantizationComparator(partitions).are_equivalent(model, extracted_model))

    def test_tomitas_7(self):
        model = WeightedTomitasGrammars.get_automaton_7()
        partitions = 10
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = WFAQuantizationComparator(partitions), max_seq_length=20)
        result = self.learner.learn(teacher, partitions = partitions, verbose = True)        
        extracted_model = result.model     
        self.assertTrue(WFAQuantizationComparator(partitions).are_equivalent(model, extracted_model))


