import unittest

from pythautomata.automata_definitions.weighted_tomitas_grammars import WeightedTomitasGrammars

from pymodelextractor.learners.observation_table_learners.bounded_pdfa_lstar_learner import BoundedPDFALStarLearner
from pythautomata.model_comparators.wfa_tolerance_comparison_strategy import WFAToleranceComparator as PDFAComparator

from pymodelextractor.teachers.pdfa_teacher import PDFATeacher

from pymodelextractor.utils.time_bound_utilities import is_unix_system


class TestBoundedPDFALStarLearner(unittest.TestCase):

    def setUp(self):
        self.comparator = PDFAComparator()
        self.learner = BoundedPDFALStarLearner(self.comparator, max_query_length=20,max_states=20)

    def test_time_bound(self):
        if is_unix_system():
            models = WeightedTomitasGrammars.get_all_automata()        
            learner = BoundedPDFALStarLearner(self.comparator, max_states= 100, max_query_length=100, max_seconds_run=60)
            for model in models:
                tolerance = 0
                teacher = PDFATeacher(model, self.comparator)
                result = learner.learn(teacher, tolerance)
                extracted_model = result.model
                self.assertEqual(model, extracted_model)
        else:
            pass

    def test_tomitas_1(self):
        model = WeightedTomitasGrammars.get_automaton_1()
        teacher = PDFATeacher(model, self.comparator)
        result = self.learner.learn(teacher)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count']>0)        
        self.assertTrue(result.info['equivalence_queries_count']>0)

    def test_tomitas_2(self):
        model = WeightedTomitasGrammars.get_automaton_2()
        teacher = PDFATeacher(model, self.comparator)
        result = self.learner.learn(teacher)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count']>0)        
        self.assertTrue(result.info['equivalence_queries_count']>0)

    def test_tomitas_3(self):
        model = WeightedTomitasGrammars.get_automaton_3()
        teacher = PDFATeacher(model, self.comparator)
        result = self.learner.learn(teacher)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count']>0)        
        self.assertTrue(result.info['equivalence_queries_count']>0)

    def test_tomitas_4(self):
        model = WeightedTomitasGrammars.get_automaton_4()
        teacher = PDFATeacher(model, self.comparator)
        result = self.learner.learn(teacher)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count']>0)        
        self.assertTrue(result.info['equivalence_queries_count']>0)

    def test_tomitas_5(self):
        model = WeightedTomitasGrammars.get_automaton_5()
        teacher = PDFATeacher(model, self.comparator)
        result = self.learner.learn(teacher)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count']>0)        
        self.assertTrue(result.info['equivalence_queries_count']>0)

    def test_tomitas_6(self):
        model = WeightedTomitasGrammars.get_automaton_6()
        teacher = PDFATeacher(model, self.comparator)
        result = self.learner.learn(teacher)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count']>0)        
        self.assertTrue(result.info['equivalence_queries_count']>0)

    def test_tomitas_7(self):
        model = WeightedTomitasGrammars.get_automaton_7()
        teacher = PDFATeacher(model, self.comparator)
        result = self.learner.learn(teacher)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count']>0)        
        self.assertTrue(result.info['equivalence_queries_count']>0)
    
    def test_bounded_tomitas_7(self):
        learner = BoundedPDFALStarLearner(self.comparator,max_states= 2, max_query_length=3)
        model = WeightedTomitasGrammars.get_automaton_7()
        teacher = PDFATeacher(model, self.comparator)
        result = learner.learn(teacher)
        extracted_model = result.model
        self.assertNotEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count']>0)        
        self.assertTrue(result.info['equivalence_queries_count']>0)
