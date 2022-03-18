import unittest
from numpy import result_type
from pythautomata.automata.wheighted_automaton_definition.weighted_state import WeightedState
from pythautomata.model_comparators.wfa_tolerance_comparison_strategy import WFAToleranceComparator
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.symbol import SymbolStr
from pythautomata.automata.wheighted_automaton_definition.probabilistic_deterministic_finite_automaton import \
    ProbabilisticDeterministicFiniteAutomaton
from pymodelextractor.learners.observation_table_learners.pdfa_lstar_learner import PDFALStarLearner
from pymodelextractor.learners.observation_table_learners.pdfa_lstarcol_learner import PDFALStarColLearner
from pymodelextractor.learners.observation_tree_learners.pdfa_quantization_n_ary_tree_learner import PDFAQuantizationNAryTreeLearner
from pythautomata.model_comparators.wfa_quantization_comparison_strategy import WFAQuantizationComparator
from pythautomata.model_comparators.wfa_tolerance_comparison_strategy import WFAToleranceComparator
from pymodelextractor.teachers.pdfa_teacher import PDFATeacher 
from pythautomata.utilities import pdfa_metrics
from pythautomata.utilities.sequence_generator import SequenceGenerator

binaryAlphabet = Alphabet(frozenset((SymbolStr('0'), SymbolStr('1'))))
zero = binaryAlphabet['0']
one = binaryAlphabet['1']


class TestWLStarVsQuant(unittest.TestCase):

    def get_example_1(self):
        q0 = WeightedState("q0", 1, 0.00)
        q1 = WeightedState("q1", 0, 0.1)
        q2 = WeightedState("q2", 0, 0.1)
        q3 = WeightedState("q3", 0, 0.1)
        q4 = WeightedState("q4", 0, 0.00)
        q5 = WeightedState("q5", 0, 0.00)
        q6 = WeightedState("q6", 0, 0.00)

        q0.add_transition(zero, q1, 0.1)
        q0.add_transition(one, q1, 0.9)
        q1.add_transition(zero, q2, 0.3)
        q1.add_transition(one, q4, 0.6)
        q2.add_transition(zero, q3, 0.3)
        q2.add_transition(one, q5, 0.6)
        q3.add_transition(zero, q3, 0.3)
        q3.add_transition(one, q6, 0.6)
        q4.add_transition(zero, q4, 0.4)
        q4.add_transition(one, q4, 0.6)
        q5.add_transition(zero, q5, 0.5)
        q5.add_transition(one, q5, 0.5)
        q6.add_transition(zero, q6, 0.6)
        q6.add_transition(one, q6, 0.4)

        states = {q0, q1, q2, q3, q4, q5, q6}
        comparator = WFAToleranceComparator()
        return ProbabilisticDeterministicFiniteAutomaton(binaryAlphabet, states, SymbolStr("$"), comparator, "example_for_paper")
    
    def get_example_2(self):
        q0 = WeightedState("q0", 1, 0.00)
        q1 = WeightedState("q1", 0, 0.1)
        q2 = WeightedState("q2", 0, 0.1)
        q3 = WeightedState("q3", 0, 0.1)
        q4 = WeightedState("q4", 0, 0.00)
        q5 = WeightedState("q5", 0, 0.00)
        q6 = WeightedState("q6", 0, 0.00)

        q0.add_transition(zero, q1, 0.1)
        q0.add_transition(one, q1, 0.9)
        q1.add_transition(zero, q2, 0.3)
        q1.add_transition(one, q4, 0.6)
        q2.add_transition(zero, q3, 0.3)
        q2.add_transition(one, q5, 0.6)
        q3.add_transition(zero, q3, 0.3)
        q3.add_transition(one, q6, 0.6)
        q4.add_transition(zero, q4, 0.5)
        q4.add_transition(one, q4, 0.5)
        q5.add_transition(zero, q5, 0.4)
        q5.add_transition(one, q5, 0.6)
        q6.add_transition(zero, q6, 0.6)
        q6.add_transition(one, q6, 0.4)

        states = {q0, q1, q2, q3, q4, q5, q6}
        comparator = WFAToleranceComparator()
        return ProbabilisticDeterministicFiniteAutomaton(binaryAlphabet, states, SymbolStr("$"), comparator, "example_for_paper")
    

    def test_1(self):
        tolerance = 0.1
        partitions = 10
        pdfa = self.get_example_1()

        WLStarColLearner = PDFALStarColLearner()
        WLStarLearner = PDFALStarLearner()
        QuantLearner = PDFAQuantizationNAryTreeLearner()

        tolerance_comparator = WFAToleranceComparator(tolerance)
        partition_comparator = WFAQuantizationComparator(partitions)

        pdfa_teacher_WLSTAR = PDFATeacher(pdfa, tolerance_comparator)
        pdfa_teacher_QUANT = PDFATeacher(pdfa, partition_comparator)

        modelWLStar = WLStarLearner.learn(pdfa_teacher_WLSTAR, tolerance).model
        modelWLStarCol = WLStarColLearner.learn(pdfa_teacher_WLSTAR, tolerance).model
        modelQUANT = QuantLearner.learn(pdfa_teacher_QUANT, partitions).model

        modelWLStar.name = 'WLSTAR_EXAMPLE'
        modelWLStarCol.name = 'WLSTAR_COL_EXAMPLE'
        modelQUANT.name = 'QUANT_EXAMPLE'

        models = [modelWLStar, modelQUANT]
        for model in models:
            model.export("./")

    def compute_stats(self, target_model, extracted_model, tolerance, partitions, test_sequences = None, sample_size = 1000, max_seq_length = 20, seed = 42):
        if test_sequences is None:
            sg = SequenceGenerator(target_model.alphabet, max_seq_length, seed)
            test_sequences = sg.generate_words(sample_size)
        
        log_probability_error = pdfa_metrics.log_probability_error(target_model, extracted_model, test_sequences)
        wer = pdfa_metrics.wer_avg(target_model, extracted_model, test_sequences)
        ndcg = pdfa_metrics.ndcg_score_avg(target_model, extracted_model, test_sequences)
        out_of_partition = pdfa_metrics.out_of_partition_elements(
            target_model, extracted_model, test_sequences, partitions)
        out_of_tolerance = pdfa_metrics.out_of_tolerance_elements(
            target_model, extracted_model, test_sequences, tolerance)
        absolute_error_avg = pdfa_metrics.absolute_error_avg(target_model, extracted_model, test_sequences)
        return log_probability_error, wer,ndcg, out_of_partition, out_of_tolerance, absolute_error_avg

    def test_2(self):
        tolerance = 0.1
        partitions = 10
        pdfa = self.get_example_2()

        WLStarColLearner = PDFALStarColLearner()
        WLStarLearner = PDFALStarLearner()
        QuantLearner = PDFAQuantizationNAryTreeLearner()

        tolerance_comparator = WFAToleranceComparator(tolerance)
        partition_comparator = WFAQuantizationComparator(partitions)

        pdfa_teacher_WLSTAR = PDFATeacher(pdfa, tolerance_comparator)
        pdfa_teacher_QUANT = PDFATeacher(pdfa, partition_comparator)

        modelWLStar = WLStarLearner.learn(pdfa_teacher_WLSTAR, tolerance).model
        #modelWLStarCol = WLStarColLearner.learn(pdfa_teacher_WLSTAR, tolerance).model
        modelQUANT = QuantLearner.learn(pdfa_teacher_QUANT, partitions).model

        modelWLStar.name = 'WLSTAR_EXAMPLE_2'
        #modelWLStarCol.name = 'WLSTAR_COL_EXAMPLE'
        modelQUANT.name = 'QUANT_EXAMPLE_2'

        models = [modelWLStar, modelQUANT]
        #for model in models:
            #model.export("./")

        max_seq_length = 100
        sample_size = 1000
        seed = 42
        sg = SequenceGenerator(pdfa.alphabet, max_seq_length, seed)
        test_sequences = sg.generate_words(sample_size)    
        log_probability_errorQ, werQ,ndcgQ, out_of_partitionQ, out_of_toleranceQ, absolute_error_avgQ = self.compute_stats(pdfa,modelQUANT, tolerance, partitions, test_sequences,None, None, 42)
        log_probability_errorW, werW,ndcgW, out_of_partitionW, out_of_toleranceW, absolute_error_avgW = self.compute_stats(pdfa,modelWLStar, tolerance, partitions, test_sequences,None, None, 42)
        print(".")

    def test_3(self):
        tolerance = 0.1
        partitions = 6
        pdfa = self.get_example_2()

        WLStarColLearner = PDFALStarColLearner()
        WLStarLearner = PDFALStarLearner()
        QuantLearner = PDFAQuantizationNAryTreeLearner()

        tolerance_comparator = WFAToleranceComparator(tolerance)
        partition_comparator = WFAQuantizationComparator(partitions)

        pdfa_teacher_WLSTAR = PDFATeacher(pdfa, tolerance_comparator)
        pdfa_teacher_QUANT = PDFATeacher(pdfa, partition_comparator)

        modelWLStar = WLStarLearner.learn(pdfa_teacher_WLSTAR, tolerance).model
        #modelWLStarCol = WLStarColLearner.learn(pdfa_teacher_WLSTAR, tolerance).model
        modelQUANT = QuantLearner.learn(pdfa_teacher_QUANT, partitions).model

        modelWLStar.name = 'WLSTAR_EXAMPLE_2'
        #modelWLStarCol.name = 'WLSTAR_COL_EXAMPLE'
        modelQUANT.name = 'QUANT_EXAMPLE_3'

        models = [modelWLStar, modelQUANT]
        #for model in models:
        #    model.export("./")
        