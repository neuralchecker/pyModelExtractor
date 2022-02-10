import unittest
from numpy import result_type

from pythautomata.automata_definitions.weighted_tomitas_grammars import WeightedTomitasGrammars

from pymodelextractor.learners.observation_table_learners.bounded_pdfa_lstar_learner import BoundedPDFALStarLearner
from pythautomata.model_comparators.wfa_tolerance_comparison_strategy import WFAToleranceComparator as PDFAComparator

from pymodelextractor.teachers.pdfa_teacher import PDFATeacher


class TestBoundedPDFALStarLearner(unittest.TestCase):

    def setUp(self):
        self.learner = BoundedPDFALStarLearner(max_query_length=20,max_states=20)

    def test_tomitas_1(self):
        model = WeightedTomitasGrammars.get_automaton_1()
        teacher = PDFATeacher(model, PDFAComparator())
        result = self.learner.learn(teacher, tolerance = 0)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count']>0)        
        self.assertTrue(result.info['equivalence_queries_count']>0)

    def test_tomitas_2(self):
        model = WeightedTomitasGrammars.get_automaton_2()
        teacher = PDFATeacher(model, PDFAComparator())
        result = self.learner.learn(teacher, tolerance = 0)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count']>0)        
        self.assertTrue(result.info['equivalence_queries_count']>0)

    def test_tomitas_3(self):
        model = WeightedTomitasGrammars.get_automaton_3()
        teacher = PDFATeacher(model, PDFAComparator())
        result = self.learner.learn(teacher, tolerance = 0)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count']>0)        
        self.assertTrue(result.info['equivalence_queries_count']>0)

    def test_tomitas_4(self):
        model = WeightedTomitasGrammars.get_automaton_4()
        teacher = PDFATeacher(model, PDFAComparator())
        result = self.learner.learn(teacher, tolerance = 0)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count']>0)        
        self.assertTrue(result.info['equivalence_queries_count']>0)

    def test_tomitas_5(self):
        model = WeightedTomitasGrammars.get_automaton_5()
        teacher = PDFATeacher(model, PDFAComparator())
        result = self.learner.learn(teacher, tolerance = 0)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count']>0)        
        self.assertTrue(result.info['equivalence_queries_count']>0)

    def test_tomitas_6(self):
        model = WeightedTomitasGrammars.get_automaton_6()
        teacher = PDFATeacher(model, PDFAComparator())
        result = self.learner.learn(teacher, tolerance = 0)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count']>0)        
        self.assertTrue(result.info['equivalence_queries_count']>0)

    def test_tomitas_7(self):
        model = WeightedTomitasGrammars.get_automaton_7()
        teacher = PDFATeacher(model, PDFAComparator())
        result = self.learner.learn(teacher, tolerance = 0)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count']>0)        
        self.assertTrue(result.info['equivalence_queries_count']>0)
    
    def test_bounded_tomitas_7(self):
        learner = BoundedPDFALStarLearner(max_states= 2, max_query_length=3)
        model = WeightedTomitasGrammars.get_automaton_7()
        teacher = PDFATeacher(model, PDFAComparator())
        result = learner.learn(teacher, tolerance = 0)
        extracted_model = result.model
        self.assertNotEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count']>0)        
        self.assertTrue(result.info['equivalence_queries_count']>0)
