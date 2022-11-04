import unittest

from pythautomata.automata_definitions.weighted_tomitas_grammars import WeightedTomitasGrammars

from pymodelextractor.learners.observation_tree_learners.pdfa_quantization_n_ary_tree_learner\
     import PDFAQuantizationNAryTreeLearner

from pymodelextractor.teachers.pac_batch_probabilistic_teacher import PACBatchProbabilisticTeacher
from pythautomata.model_comparators.wfa_quantization_comparison_strategy import WFAQuantizationComparator


class TestPACBatchTeacherQuant(unittest.TestCase):

    def setUp(self):
        self.partitions = 10
        self.comparator = WFAQuantizationComparator(self.partitions)
        self.learner = PDFAQuantizationNAryTreeLearner(self.comparator)

    def test_tomitas_1(self):
        model = WeightedTomitasGrammars.get_automaton_1()
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = self.comparator,
                                               max_seq_length=20)
        result = self.learner.learn(teacher, verbose = True)        
        extracted_model = result.model     
        self.assertTrue(self.comparator.are_equivalent(model, extracted_model))
    
    def test_tomitas_2(self):
        model = WeightedTomitasGrammars.get_automaton_2()
        
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = self.comparator,
                                               max_seq_length=20)
        result = self.learner.learn(teacher, verbose = True)        
        extracted_model = result.model     
        self.assertTrue(self.comparator.are_equivalent(model, extracted_model))
    

    def test_tomitas_3(self):
        model = WeightedTomitasGrammars.get_automaton_3()
        
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = self.comparator,
                                               max_seq_length=20)
        result = self.learner.learn(teacher, verbose = True)        
        extracted_model = result.model     
        self.assertTrue(self.comparator.are_equivalent(model, extracted_model))

    def test_tomitas_4(self):
        model = WeightedTomitasGrammars.get_automaton_4()
        
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = self.comparator,
                                               max_seq_length=20)
        result = self.learner.learn(teacher, verbose = True)        
        extracted_model = result.model     
        self.assertTrue(self.comparator.are_equivalent(model, extracted_model))
    
    def test_tomitas_5(self):
        model = WeightedTomitasGrammars.get_automaton_5()
        
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = self.comparator,
                                               max_seq_length=20)
        result = self.learner.learn(teacher, verbose = True)        
        extracted_model = result.model     
        self.assertTrue(self.comparator.are_equivalent(model, extracted_model))
    

    def test_tomitas_6(self):
        model = WeightedTomitasGrammars.get_automaton_6()
        
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = self.comparator,
                                               max_seq_length=20)
        result = self.learner.learn(teacher, verbose = True)        
        extracted_model = result.model     
        self.assertTrue(self.comparator.are_equivalent(model, extracted_model))

    def test_tomitas_7(self):
        model = WeightedTomitasGrammars.get_automaton_7()
        
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = self.comparator,
                                               max_seq_length=20)
        result = self.learner.learn(teacher, verbose = True)        
        extracted_model = result.model     
        self.assertTrue(self.comparator.are_equivalent(model, extracted_model))


