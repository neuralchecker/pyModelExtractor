import unittest

from pythautomata.automata_definitions.weighted_tomitas_grammars import WeightedTomitasGrammars

from pymodelextractor.learners.observation_table_learners.pdfa_lstar_learner import PDFALStarLearner
from pymodelextractor.teachers.pdfa_teacher import PDFATeacher


class TestPDFALStarLearner(unittest.TestCase):

    def setUp(self):
        self.learner = PDFALStarLearner()

    def test_tomitas_1(self):
        model = WeightedTomitasGrammars.get_automaton_1()
        teacher = PDFATeacher(model, 0)
        extracted_model = self.learner.learn(teacher).model
        self.assertEqual(model, extracted_model)

    def test_tomitas_2(self):
        model = WeightedTomitasGrammars.get_automaton_2()
        teacher = PDFATeacher(model, 0)
        extracted_model = self.learner.learn(teacher).model
        self.assertEqual(model, extracted_model)

    def test_tomitas_3(self):
        model = WeightedTomitasGrammars.get_automaton_3()
        teacher = PDFATeacher(model, 0)
        extracted_model = self.learner.learn(teacher).model
        self.assertEqual(model, extracted_model)

    def test_tomitas_4(self):
        model = WeightedTomitasGrammars.get_automaton_4()
        teacher = PDFATeacher(model, 0)
        extracted_model = self.learner.learn(teacher).model
        self.assertEqual(model, extracted_model)

    def test_tomitas_5(self):
        model = WeightedTomitasGrammars.get_automaton_5()
        teacher = PDFATeacher(model, 0)
        extracted_model = self.learner.learn(teacher).model
        self.assertEqual(model, extracted_model)

    def test_tomitas_6(self):
        model = WeightedTomitasGrammars.get_automaton_6()
        teacher = PDFATeacher(model, 0)
        extracted_model = self.learner.learn(teacher).model
        self.assertEqual(model, extracted_model)

    def test_tomitas_7(self):
        model = WeightedTomitasGrammars.get_automaton_7()
        teacher = PDFATeacher(model, 0)
        extracted_model = self.learner.learn(teacher).model
        self.assertEqual(model, extracted_model)
