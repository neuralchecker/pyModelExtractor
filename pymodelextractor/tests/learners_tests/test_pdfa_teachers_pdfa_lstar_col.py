import unittest
from numpy import result_type

from pythautomata.automata_definitions.weighted_tomitas_grammars import WeightedTomitasGrammars

from pymodelextractor.learners.observation_table_learners.pdfa_lstarcol_learner import PDFALStarColLearner
from pythautomata.model_comparators.wfa_comparison_strategy import WFAComparator as PDFAComparator

from pymodelextractor.teachers.pdfa_teacher import PDFATeacher
from pymodelextractor.teachers.sample_probabilistic_teacher import SampleProbabilisticTeacher
from pymodelextractor.teachers.pac_probabilistic_teacher import PACProbabilisticTeacher

class TestPDFATeachersLStarCol(unittest.TestCase):

    def setUp(self):
        self.learner = PDFALStarColLearner()

    def test_tomitas_1(self):
        model = WeightedTomitasGrammars.get_automaton_1()
        teacher1 = PDFATeacher(model, 0, PDFAComparator())
        teacher2 = SampleProbabilisticTeacher(model, 0, sample_size=500, max_seq_length=20)
        teacher3 = PACProbabilisticTeacher(model, 0.05, 0.01, 0, max_seq_length=20)
        result1 = self.learner.learn(teacher1, verbose = True)
        result2 = self.learner.learn(teacher2, verbose = True)
        result3 = self.learner.learn(teacher3, verbose = True)
        extracted_model1 = result1.model
        extracted_model2 = result2.model
        extracted_model3 = result3.model
        self.assertEqual(model, extracted_model1)
        self.assertEqual(model, extracted_model2)
        self.assertEqual(model, extracted_model3)

    def test_tomitas_2(self):
        model = WeightedTomitasGrammars.get_automaton_2()
        teacher1 = PDFATeacher(model, 0, PDFAComparator())
        teacher2 = SampleProbabilisticTeacher(model, 0, sample_size=500, max_seq_length=20)
        teacher3 = PACProbabilisticTeacher(model, 0.05, 0.01, 0, max_seq_length=20)
        result1 = self.learner.learn(teacher1, verbose = True)
        result2 = self.learner.learn(teacher2, verbose = True)
        result3 = self.learner.learn(teacher3, verbose = True)
        extracted_model1 = result1.model
        extracted_model2 = result2.model
        extracted_model3 = result3.model
        self.assertEqual(model, extracted_model1)
        self.assertEqual(model, extracted_model2)
        self.assertEqual(model, extracted_model3)

    def test_tomitas_3(self):
        model = WeightedTomitasGrammars.get_automaton_3()
        teacher1 = PDFATeacher(model, 0, PDFAComparator())
        teacher2 = SampleProbabilisticTeacher(model, 0, sample_size=500, max_seq_length=20)
        teacher3 = PACProbabilisticTeacher(model, 0.05, 0.01, 0, max_seq_length=20)
        result1 = self.learner.learn(teacher1, verbose = True)
        result2 = self.learner.learn(teacher2, verbose = True)
        result3 = self.learner.learn(teacher3, verbose = True)
        extracted_model1 = result1.model
        extracted_model2 = result2.model
        extracted_model3 = result3.model
        self.assertEqual(model, extracted_model1)
        self.assertEqual(model, extracted_model2)
        self.assertEqual(model, extracted_model3)
        
    def test_tomitas_4(self):
        model = WeightedTomitasGrammars.get_automaton_4()
        teacher1 = PDFATeacher(model, 0, PDFAComparator())
        teacher2 = SampleProbabilisticTeacher(model, 0, sample_size=500, max_seq_length=20)
        teacher3 = PACProbabilisticTeacher(model, 0.05, 0.01, 0, max_seq_length=20)
        result1 = self.learner.learn(teacher1, verbose = True)
        result2 = self.learner.learn(teacher2, verbose = True)
        result3 = self.learner.learn(teacher3, verbose = True)
        extracted_model1 = result1.model
        extracted_model2 = result2.model
        extracted_model3 = result3.model        
        self.assertEqual(model, extracted_model1)
        self.assertEqual(model, extracted_model2)
        self.assertEqual(model, extracted_model3)
        

    def test_tomitas_5(self):
        model = WeightedTomitasGrammars.get_automaton_5()
        teacher1 = PDFATeacher(model, 0, PDFAComparator())
        teacher2 = SampleProbabilisticTeacher(model, 0, sample_size=500, max_seq_length=20)
        teacher3 = PACProbabilisticTeacher(model, 0.05, 0.01, 0, max_seq_length=20)
        result1 = self.learner.learn(teacher1, verbose = True)
        result2 = self.learner.learn(teacher2, verbose = True)
        result3 = self.learner.learn(teacher3, verbose = True)
        extracted_model1 = result1.model
        extracted_model2 = result2.model
        extracted_model3 = result3.model        
        self.assertEqual(model, extracted_model1)
        self.assertEqual(model, extracted_model2)
        self.assertEqual(model, extracted_model3)

    def test_tomitas_6(self):
        model = WeightedTomitasGrammars.get_automaton_6()
        teacher1 = PDFATeacher(model, 0, PDFAComparator())
        teacher2 = SampleProbabilisticTeacher(model, 0, sample_size=500, max_seq_length=20)
        teacher3 = PACProbabilisticTeacher(model, 0.05, 0.01, 0, max_seq_length=20)
        result1 = self.learner.learn(teacher1, verbose = True)
        result2 = self.learner.learn(teacher2, verbose = True)
        result3 = self.learner.learn(teacher3, verbose = True)
        extracted_model1 = result1.model
        extracted_model2 = result2.model
        extracted_model3 = result3.model
        self.assertEqual(model, extracted_model1)
        self.assertEqual(model, extracted_model2)
        self.assertEqual(model, extracted_model3)

    def test_tomitas_7(self):
        model = WeightedTomitasGrammars.get_automaton_7()
        teacher1 = PDFATeacher(model, 0, PDFAComparator())
        teacher2 = SampleProbabilisticTeacher(model, 0, sample_size=500, max_seq_length=20)
        teacher3 = PACProbabilisticTeacher(model, 0.05, 0.01, 0, max_seq_length=20)
        result1 = self.learner.learn(teacher1, verbose = True)
        result2 = self.learner.learn(teacher2, verbose = True)
        result3 = self.learner.learn(teacher3, verbose = True)
        extracted_model1 = result1.model
        extracted_model2 = result2.model
        extracted_model3 = result3.model
        self.assertEqual(model, extracted_model1)
        self.assertEqual(model, extracted_model2)
        self.assertEqual(model, extracted_model3)
