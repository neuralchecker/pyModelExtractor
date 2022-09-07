import unittest

from pythautomata.automata_definitions.weighted_tomitas_grammars import WeightedTomitasGrammars

from pymodelextractor.learners.observation_tree_learners.pdfa_quantization_n_ary_tree_learner import PDFAQuantizationNAryTreeLearner
from pymodelextractor.learners.observation_table_learners.pdfa_lstar_learner import PDFALStarLearner
from pythautomata.model_comparators.wfa_tolerance_comparison_strategy import WFAToleranceComparator
from pythautomata.model_comparators.wfa_quantization_comparison_strategy import WFAQuantizationComparator

from pymodelextractor.teachers.pdfa_teacher import PDFATeacher

from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.symbol import SymbolStr
from pythautomata.utilities.sequence_generator import SequenceGenerator
from pythautomata.utilities import pdfa_metrics

binaryAlphabet = Alphabet(frozenset((SymbolStr('0'), SymbolStr('1'))))


class TestPDFAQuantizantionNAryTreeLearnerMetrics(unittest.TestCase):

    def setUp(self):
        self.partitions = 10
        self.quant_comparator = WFAQuantizationComparator(self.partitions)
        self.tolerance_comparator = WFAToleranceComparator(1/self.partitions)

        self.QuaNTLearner = PDFAQuantizationNAryTreeLearner(self.quant_comparator)
        self.WLSstarLearner = PDFALStarLearner(self.tolerance_comparator)

    def test_tomitas_1(self):
        model = WeightedTomitasGrammars.get_automaton_1()        
        teacher1 = PDFATeacher(model, self.quant_comparator)        
        teacher2 = PDFATeacher(model, self.tolerance_comparator)
        result1 = self.QuaNTLearner.learn(teacher1)
        result2 = self.WLSstarLearner.learn(teacher2)
        model1 = result1.model
        model2 = result2.model
        model1.name = 'res1'
        model2.name = 'res2'
        self.assertEqual(model, model1)
        self.assertEqual(model1, model)
        self.assertEqual(model2, model)
        self.assertTrue(result1.info['last_token_weight_queries_count']>0)        
        self.assertTrue(result1.info['equivalence_queries_count']>0)

        sg = SequenceGenerator(model.alphabet, 20, 42)
        test_sequences = sg.generate_words(500)
        m1 = pdfa_metrics.log_probability_error(model, model1, test_sequences)
        m2 = pdfa_metrics.wer_avg(model, model1, test_sequences)
        m3 = pdfa_metrics.ndcg_score_avg(model, model1, test_sequences)
        m4 = pdfa_metrics.out_of_partition_elements(
            model, model1, test_sequences, 10)
        m5 = pdfa_metrics.out_of_tolerance_elements(
            model, model1, test_sequences, 0.1)
        self.assertEqual(m1, 0)
        self.assertEqual(m2, 0)
        self.assertEqual(m3, 1)
        self.assertEqual(m4, 0)
        self.assertEqual(m5, 0)



    