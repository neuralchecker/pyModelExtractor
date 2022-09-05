import unittest

from pythautomata.automata_definitions.weighted_tomitas_grammars import WeightedTomitasGrammars

from pymodelextractor.factories.pdfa_extraction_factory import PDFAExtractionFactory


class TestPDFALStarColQuantLearner(unittest.TestCase):

    def setUp(self):
        self.factory = PDFAExtractionFactory()

    def test_tomitas_1(self):
        model = WeightedTomitasGrammars.get_automaton_1()
        learner, teacher = self.factory.probabilistic_lstarcol_quant_extraction(model, 10)
        result = learner.learn(teacher)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count'] > 0)
        self.assertTrue(result.info['equivalence_queries_count'] > 0)

    def test_tomitas_2(self):
        model = WeightedTomitasGrammars.get_automaton_2()
        learner, teacher = self.factory.probabilistic_lstarcol_quant_extraction(model, 10)
        result = learner.learn(teacher)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count'] > 0)
        self.assertTrue(result.info['equivalence_queries_count'] > 0)

    def test_tomitas_3(self):
        model = WeightedTomitasGrammars.get_automaton_3()
        learner, teacher = self.factory.probabilistic_lstarcol_quant_extraction(model, 10)
        result = learner.learn(teacher)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count'] > 0)
        self.assertTrue(result.info['equivalence_queries_count'] > 0)

    def test_tomitas_4(self):
        model = WeightedTomitasGrammars.get_automaton_4()
        learner, teacher = self.factory.probabilistic_lstarcol_quant_extraction(model, 10)
        result = learner.learn(teacher)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count'] > 0)
        self.assertTrue(result.info['equivalence_queries_count'] > 0)

    def test_tomitas_5(self):
        model = WeightedTomitasGrammars.get_automaton_5()
        learner, teacher = self.factory.probabilistic_lstarcol_quant_extraction(model, 10)
        result = learner.learn(teacher)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count'] > 0)
        self.assertTrue(result.info['equivalence_queries_count'] > 0)

    def test_tomitas_6(self):
        model = WeightedTomitasGrammars.get_automaton_6()
        learner, teacher = self.factory.probabilistic_lstarcol_quant_extraction(model, 10)
        result = learner.learn(teacher)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count'] > 0)
        self.assertTrue(result.info['equivalence_queries_count'] > 0)

    def test_tomitas_7(self):
        model = WeightedTomitasGrammars.get_automaton_7()
        learner, teacher = self.factory.probabilistic_lstarcol_quant_extraction(model, 10)
        result = learner.learn(teacher)
        extracted_model = result.model
        self.assertEqual(model, extracted_model)
        self.assertTrue(result.info['last_token_weight_queries_count'] > 0)
        self.assertTrue(result.info['equivalence_queries_count'] > 0)
