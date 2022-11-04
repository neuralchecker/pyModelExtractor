import unittest

from pythautomata.automata_definitions.weighted_tomitas_grammars import WeightedTomitasGrammars

from pymodelextractor.learners.observation_table_learners.pdfa_lstar_learner import PDFALStarLearner

from pymodelextractor.teachers.pdfa_teacher import PDFATeacher
from pymodelextractor.teachers.sample_probabilistic_teacher import SampleProbabilisticTeacher
from pymodelextractor.teachers.pac_probabilistic_teacher import PACProbabilisticTeacher
from pythautomata.model_comparators.wfa_tolerance_comparison_strategy import WFAToleranceComparator
from pymodelextractor.teachers.sample_batch_probabilistic_teacher import SampleBatchProbabilisticTeacher


class TestPDFATeachersLStar(unittest.TestCase):

    def setUp(self):
        self.learner = PDFALStarLearner()

    def test_tomitas_1(self):
        model = WeightedTomitasGrammars.get_automaton_1()
        teacher1 = PDFATeacher(model, WFAToleranceComparator())
        teacher2 = SampleProbabilisticTeacher(model, comparator=WFAToleranceComparator(), sample_size=500, 
                                              max_seq_length=20)                    
        teacher3 = PACProbabilisticTeacher(model, epsilon = 0.05, delta = 0.01, comparator=WFAToleranceComparator(), max_seq_length=20)
        teacher4 = SampleProbabilisticTeacher(model, comparator=WFAToleranceComparator(), full_prefix_set=True, 
                                              max_seq_length=10)
        teacher5 = SampleBatchProbabilisticTeacher(model, comparator=WFAToleranceComparator(), full_prefix_set=True, 
                                              max_seq_length=10)
        result1 = self.learner.learn(teacher1, verbose=True)
        result2 = self.learner.learn(teacher2, verbose=True)
        result3 = self.learner.learn(teacher3, verbose=True)
        result4 = self.learner.learn(teacher4, verbose=True)  
        result5 = self.learner.learn(teacher5, verbose=True)   
        extracted_model1 = result1.model
        extracted_model2 = result2.model
        extracted_model3 = result3.model        
        extracted_model4 = result4.model      
        extracted_model5 = result5.model
        self.assertEqual(model, extracted_model1)
        self.assertTrue(WFAToleranceComparator(0.000001).are_equivalent(model, extracted_model2))        
        self.assertTrue(WFAToleranceComparator(0.000001).are_equivalent(model, extracted_model3))
        self.assertEqual(model, extracted_model3)
        self.assertTrue(WFAToleranceComparator(0.000001).are_equivalent(model, extracted_model4))
        self.assertTrue(WFAToleranceComparator(0.000001).are_equivalent(model, extracted_model5))

    def test_tomitas_2(self):
        model = WeightedTomitasGrammars.get_automaton_2()
        teacher1 = PDFATeacher(model, WFAToleranceComparator())
        teacher2 = SampleProbabilisticTeacher(model, comparator=WFAToleranceComparator(), sample_size=500, 
                                              max_seq_length=20)
        teacher3 = PACProbabilisticTeacher(model, epsilon = 0.05, delta = 0.01, comparator=WFAToleranceComparator(), max_seq_length=20)
        teacher4 = SampleProbabilisticTeacher(model, comparator=WFAToleranceComparator(), full_prefix_set=True, 
                                              max_seq_length=10)
        teacher5 = SampleBatchProbabilisticTeacher(model, comparator=WFAToleranceComparator(), full_prefix_set=True, 
                                              max_seq_length=10)                                      
        result1 = self.learner.learn(teacher1, verbose=True)
        result2 = self.learner.learn(teacher2, verbose=True)
        result3 = self.learner.learn(teacher3, verbose=True)        
        result4 = self.learner.learn(teacher4, verbose=True)                
        result5 = self.learner.learn(teacher5, verbose=True)   
        extracted_model1 = result1.model
        extracted_model2 = result2.model
        extracted_model3 = result3.model
        extracted_model4 = result4.model        
        extracted_model5 = result5.model
        self.assertEqual(model, extracted_model1)
        self.assertTrue(WFAToleranceComparator(0.000001).are_equivalent(model, extracted_model2))        
        self.assertTrue(WFAToleranceComparator(0.000001).are_equivalent(model, extracted_model3))
        self.assertEqual(model, extracted_model3)
        self.assertTrue(WFAToleranceComparator(0.000001).are_equivalent(model, extracted_model4))        
        self.assertTrue(WFAToleranceComparator(0.000001).are_equivalent(model, extracted_model5))


    def test_tomitas_3(self):
        model = WeightedTomitasGrammars.get_automaton_3()
        tolerance = 0.000001
        teacher1 = PDFATeacher(model, WFAToleranceComparator(tolerance))
        teacher2 = SampleProbabilisticTeacher(model, comparator = WFAToleranceComparator(tolerance), sample_size=500,
                                              max_seq_length=20)
        teacher3 = PACProbabilisticTeacher(model, epsilon = 0.05, delta = 0.01, comparator = WFAToleranceComparator(tolerance),
                                           max_seq_length=20)
        teacher4 = SampleProbabilisticTeacher(model, comparator=WFAToleranceComparator(tolerance), full_prefix_set=True, 
                                              max_seq_length=10)
        teacher5 = SampleBatchProbabilisticTeacher(model, comparator=WFAToleranceComparator(), full_prefix_set=True, 
                                              max_seq_length=10)
        result1 = self.learner.learn(teacher1, verbose=True)
        result2 = self.learner.learn(teacher2, verbose=True)
        result3 = self.learner.learn(teacher3, verbose=True)
        result4 = self.learner.learn(teacher4, verbose=True)
        result5 = self.learner.learn(teacher5, verbose=True)
        extracted_model1 = result1.model
        extracted_model2 = result2.model
        extracted_model3 = result3.model
        extracted_model4 = result4.model
        extracted_model5 = result5.model
        self.assertEqual(model, extracted_model1)
        self.assertTrue(WFAToleranceComparator(tolerance).are_equivalent(model, extracted_model2))        
        self.assertTrue(WFAToleranceComparator(tolerance).are_equivalent(model, extracted_model3))    
        self.assertTrue(WFAToleranceComparator(tolerance).are_equivalent(model, extracted_model4))
        self.assertTrue(WFAToleranceComparator(tolerance).are_equivalent(model, extracted_model5))


    def test_tomitas_4(self):
        model = WeightedTomitasGrammars.get_automaton_4()
        tolerance = 0.000001
        teacher1 = PDFATeacher(model, WFAToleranceComparator(tolerance))        
        teacher2 = SampleProbabilisticTeacher(model, comparator=WFAToleranceComparator(tolerance), sample_size=500,
                                              max_seq_length=20)
        teacher3 = PACProbabilisticTeacher(model, epsilon = 0.05, delta = 0.01, comparator=WFAToleranceComparator(tolerance),
                                           max_seq_length=20)
        teacher4 = SampleProbabilisticTeacher(model, comparator=WFAToleranceComparator(tolerance), full_prefix_set=True, 
                                              max_seq_length=10)
        teacher5 = SampleBatchProbabilisticTeacher(model, comparator=WFAToleranceComparator(), full_prefix_set=True, 
                                              max_seq_length=10)
        result1 = self.learner.learn(teacher1, verbose=True)
        result2 = self.learner.learn(teacher2, verbose=True)
        result3 = self.learner.learn(teacher3, verbose=True)
        result4 = self.learner.learn(teacher4, verbose=True)
        result5 = self.learner.learn(teacher5, verbose=True)
        extracted_model1 = result1.model
        extracted_model2 = result2.model
        extracted_model3 = result3.model
        extracted_model4 = result4.model
        extracted_model5 = result5.model
        self.assertEqual(model, extracted_model1)
        self.assertTrue(WFAToleranceComparator(tolerance).are_equivalent(model, extracted_model2))        
        self.assertTrue(WFAToleranceComparator(tolerance).are_equivalent(model, extracted_model3))    
        self.assertTrue(WFAToleranceComparator(tolerance).are_equivalent(model, extracted_model4))
        self.assertTrue(WFAToleranceComparator(tolerance).are_equivalent(model, extracted_model5))

        

    def test_tomitas_5(self):
        model = WeightedTomitasGrammars.get_automaton_5()
        tolerance = 0.000001
        teacher1 = PDFATeacher(model, WFAToleranceComparator(tolerance))        
        teacher2 = SampleProbabilisticTeacher(model, comparator=WFAToleranceComparator(tolerance), sample_size=500,
                                              max_seq_length=20)
        teacher3 = PACProbabilisticTeacher(model, epsilon = 0.05, delta = 0.01, comparator=WFAToleranceComparator(tolerance),
                                           max_seq_length=20)
        teacher4 = SampleProbabilisticTeacher(model, comparator=WFAToleranceComparator(tolerance), full_prefix_set=True, 
                                              max_seq_length=10)
        teacher5 = SampleBatchProbabilisticTeacher(model, comparator=WFAToleranceComparator(), full_prefix_set=True, 
                                              max_seq_length=10)
        result1 = self.learner.learn(teacher1, verbose=True)
        result2 = self.learner.learn(teacher2, verbose=True)
        result3 = self.learner.learn(teacher3, verbose=True)
        result4 = self.learner.learn(teacher4, verbose=True)
        result5 = self.learner.learn(teacher5, verbose=True)
        extracted_model1 = result1.model
        extracted_model2 = result2.model
        extracted_model3 = result3.model
        extracted_model4 = result4.model
        extracted_model5 = result5.model
        self.assertEqual(model, extracted_model1)
        self.assertTrue(WFAToleranceComparator(tolerance).are_equivalent(model, extracted_model2))        
        self.assertTrue(WFAToleranceComparator(tolerance).are_equivalent(model, extracted_model3))    
        self.assertTrue(WFAToleranceComparator(tolerance).are_equivalent(model, extracted_model4))
        self.assertTrue(WFAToleranceComparator(tolerance).are_equivalent(model, extracted_model5))

    def test_tomitas_6(self):
        model = WeightedTomitasGrammars.get_automaton_6()
        tolerance = 0.000001
        teacher1 = PDFATeacher(model, WFAToleranceComparator(tolerance))        
        teacher2 = SampleProbabilisticTeacher(model, comparator=WFAToleranceComparator(tolerance), sample_size=500,
                                              max_seq_length=20)
        teacher3 = PACProbabilisticTeacher(model, epsilon = 0.05, delta = 0.01, comparator = WFAToleranceComparator(tolerance),
                                           max_seq_length=20)
        teacher4 = SampleProbabilisticTeacher(model, comparator=WFAToleranceComparator(tolerance), full_prefix_set=True, 
                                              max_seq_length=10)
        teacher5 = SampleBatchProbabilisticTeacher(model, comparator=WFAToleranceComparator(), full_prefix_set=True, 
                                              max_seq_length=10)
        result1 = self.learner.learn(teacher1, verbose=True)
        result2 = self.learner.learn(teacher2, verbose=True)
        result3 = self.learner.learn(teacher3, verbose=True)
        result4 = self.learner.learn(teacher4, verbose=True)
        result5 = self.learner.learn(teacher5, verbose=True)
        extracted_model1 = result1.model
        extracted_model2 = result2.model
        extracted_model3 = result3.model
        extracted_model4 = result4.model
        extracted_model5 = result5.model
        self.assertEqual(model, extracted_model1)
        self.assertTrue(WFAToleranceComparator(tolerance).are_equivalent(model, extracted_model2))        
        self.assertTrue(WFAToleranceComparator(tolerance).are_equivalent(model, extracted_model3))    
        self.assertTrue(WFAToleranceComparator(tolerance).are_equivalent(model, extracted_model4))
        self.assertTrue(WFAToleranceComparator(tolerance).are_equivalent(model, extracted_model5))

        
    def test_tomitas_7(self):
        model = WeightedTomitasGrammars.get_automaton_7()
        tolerance = 0.000001
        teacher1 = PDFATeacher(model, WFAToleranceComparator(tolerance))        
        teacher2 = SampleProbabilisticTeacher(model, comparator=WFAToleranceComparator(tolerance), sample_size=500,
                                              max_seq_length=20)
        teacher3 = PACProbabilisticTeacher(model, epsilon = 0.05, delta = 0.01, comparator = WFAToleranceComparator(tolerance),
                                           max_seq_length=20)
        teacher4 = SampleProbabilisticTeacher(model, comparator=WFAToleranceComparator(tolerance), full_prefix_set=True, 
                                              max_seq_length=10)
        teacher5 = SampleBatchProbabilisticTeacher(model, comparator=WFAToleranceComparator(), full_prefix_set=True, 
                                              max_seq_length=10)
        result1 = self.learner.learn(teacher1, verbose=True)
        result2 = self.learner.learn(teacher2, verbose=True)
        result3 = self.learner.learn(teacher3, verbose=True)
        result4 = self.learner.learn(teacher4, verbose=True)
        result5 = self.learner.learn(teacher5, verbose=True)
        extracted_model1 = result1.model
        extracted_model2 = result2.model
        extracted_model3 = result3.model
        extracted_model4 = result4.model
        extracted_model5 = result5.model
        self.assertEqual(model, extracted_model1)
        self.assertTrue(WFAToleranceComparator(tolerance).are_equivalent(model, extracted_model2))        
        self.assertTrue(WFAToleranceComparator(tolerance).are_equivalent(model, extracted_model3))    
        self.assertTrue(WFAToleranceComparator(tolerance).are_equivalent(model, extracted_model4))
        self.assertTrue(WFAToleranceComparator(tolerance).are_equivalent(model, extracted_model5))
