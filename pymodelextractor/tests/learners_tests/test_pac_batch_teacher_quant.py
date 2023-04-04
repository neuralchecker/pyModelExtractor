import unittest

from pythautomata.automata_definitions.weighted_tomitas_grammars import WeightedTomitasGrammars

from pymodelextractor.learners.observation_tree_learners.pdfa_quantization_n_ary_tree_learner\
     import PDFAQuantizationNAryTreeLearner

from pymodelextractor.teachers.pac_batch_probabilistic_teacher import PACBatchProbabilisticTeacher
from pythautomata.model_comparators.wfa_quantization_comparison_strategy import WFAQuantizationComparator
from pythautomata.model_comparators.wfa_partition_comparison_strategy import WFAPartitionComparator
from pythautomata.utilities.probability_partitioner import QuantizationProbabilityPartitioner
from pythautomata.utilities.uniform_length_sequence_generator import UniformLengthSequenceGenerator

from pymodelextractor.utils.pickle_data_loader import PickleDataLoader

from collections import OrderedDict
import joblib
import os

class TestPACBatchTeacherQuant(unittest.TestCase):

    def setUp(self):
        self.partitions = 10
        self.comparator = WFAQuantizationComparator(self.partitions)
        self.probability_partitioner = QuantizationProbabilityPartitioner(self.partitions)
        self.comparator = WFAPartitionComparator(self.probability_partitioner)
        self.learner = PDFAQuantizationNAryTreeLearner

    def test_tomitas_1(self):
        model = WeightedTomitasGrammars.get_automaton_1()
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = self.comparator,
                                               max_seq_length=20)
        result = self.learner(self.probability_partitioner).learn(teacher, verbose = True)        
        extracted_model = result.model     
        self.assertTrue(self.comparator.are_equivalent(model, extracted_model))
    
    def test_tomitas_2(self):
        model = WeightedTomitasGrammars.get_automaton_2()
        
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = self.comparator,
                                               max_seq_length=20)
        result = self.learner(self.probability_partitioner).learn(teacher, verbose = True)        
        extracted_model = result.model     
        self.assertTrue(self.comparator.are_equivalent(model, extracted_model))
    

    def test_tomitas_3(self):
        model = WeightedTomitasGrammars.get_automaton_3()
        
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = self.comparator,
                                               max_seq_length=20)
        result = self.learner(self.probability_partitioner).learn(teacher, verbose = True)        
        extracted_model = result.model     
        self.assertTrue(self.comparator.are_equivalent(model, extracted_model))

    def test_tomitas_4(self):
        model = WeightedTomitasGrammars.get_automaton_4()
        
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = self.comparator,
                                               max_seq_length=20)
        result = self.learner(self.probability_partitioner).learn(teacher, verbose = True)        
        extracted_model = result.model     
        self.assertTrue(self.comparator.are_equivalent(model, extracted_model))
    
    def test_tomitas_5(self):
        model = WeightedTomitasGrammars.get_automaton_5()
        
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = self.comparator,
                                               max_seq_length=20)
        result = self.learner(self.probability_partitioner).learn(teacher, verbose = True)        
        extracted_model = result.model     
        self.assertTrue(self.comparator.are_equivalent(model, extracted_model))
    

    def test_tomitas_6(self):
        model = WeightedTomitasGrammars.get_automaton_6()
        
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = self.comparator,
                                               max_seq_length=20)
        result = self.learner(self.probability_partitioner).learn(teacher, verbose = True)        
        extracted_model = result.model     
        self.assertTrue(self.comparator.are_equivalent(model, extracted_model))

    def test_tomitas_7(self):
        model = WeightedTomitasGrammars.get_automaton_7()
        
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = self.comparator,
                                               max_seq_length=20)
        result = self.learner(self.probability_partitioner).learn(teacher, verbose = True)        
        extracted_model = result.model     
        self.assertTrue(self.comparator.are_equivalent(model, extracted_model))


    def test_tomitas_1_w_pre_cache(self):
        model = WeightedTomitasGrammars.get_automaton_1()
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = self.comparator,
                                               max_seq_length=20)
        result = self.learner(self.probability_partitioner, pre_cache_queries_for_building_hipothesis = True).learn(teacher, verbose = True)        
        extracted_model = result.model     
        self.assertTrue(self.comparator.are_equivalent(model, extracted_model))
    
    def test_tomitas_2_w_pre_cache(self):
        model = WeightedTomitasGrammars.get_automaton_2()
        
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = self.comparator,
                                               max_seq_length=20)
        result = self.learner(self.probability_partitioner, pre_cache_queries_for_building_hipothesis = True).learn(teacher, verbose = True)           
        extracted_model = result.model     
        self.assertTrue(self.comparator.are_equivalent(model, extracted_model))
    

    def test_tomitas_3_w_pre_cache(self):
        model = WeightedTomitasGrammars.get_automaton_3()
        
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = self.comparator,
                                               max_seq_length=20)
        result = self.learner(self.probability_partitioner, pre_cache_queries_for_building_hipothesis = True).learn(teacher, verbose = True)           
        extracted_model = result.model     
        self.assertTrue(self.comparator.are_equivalent(model, extracted_model))

    def test_tomitas_4_w_pre_cache(self):
        model = WeightedTomitasGrammars.get_automaton_4()
        
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = self.comparator,
                                               max_seq_length=20)
        result = self.learner(self.probability_partitioner, pre_cache_queries_for_building_hipothesis = True).learn(teacher, verbose = True)     
        extracted_model = result.model     
        self.assertTrue(self.comparator.are_equivalent(model, extracted_model))
    
    def test_tomitas_5_w_pre_cache(self):
        model = WeightedTomitasGrammars.get_automaton_5()
        
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = self.comparator,
                                               max_seq_length=20)
        result = self.learner(self.probability_partitioner, pre_cache_queries_for_building_hipothesis = True).learn(teacher, verbose = True)     
        extracted_model = result.model     
        self.assertTrue(self.comparator.are_equivalent(model, extracted_model))
    

    def test_tomitas_6_w_pre_cache(self):
        model = WeightedTomitasGrammars.get_automaton_6()
        
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = self.comparator,
                                               max_seq_length=20)
        result = self.learner(self.probability_partitioner, pre_cache_queries_for_building_hipothesis = True).learn(teacher, verbose = True)     
        extracted_model = result.model     
        self.assertTrue(self.comparator.are_equivalent(model, extracted_model))

    def test_tomitas_7_w_pre_cache(self):
        model = WeightedTomitasGrammars.get_automaton_7()
        
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = self.comparator,
                                               max_seq_length=20)
        result = self.learner(self.probability_partitioner, pre_cache_queries_for_building_hipothesis = True).learn(teacher, verbose = True)            
        extracted_model = result.model     
        self.assertTrue(self.comparator.are_equivalent(model, extracted_model))


    def test_tomitas_1_w_parallel_cache(self):
        model = WeightedTomitasGrammars.get_automaton_1()
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = self.comparator,
                                               max_seq_length=20,parallel_cache=True)
        result = self.learner(self.probability_partitioner, pre_cache_queries_for_building_hipothesis = True).learn(teacher, verbose = True)        
        extracted_model = result.model     
        self.assertTrue(self.comparator.are_equivalent(model, extracted_model))
    
    
    def test_tomitas_1_w_data_loader_cache_and_parallel_cache(self):
        model = WeightedTomitasGrammars.get_automaton_1()
        generator = UniformLengthSequenceGenerator(alphabet=model.alphabet, max_seq_length=5, min_seq_length=0)
        sequences = generator.generate_words(2000)
        path = "./test_dataset"
        symbols = list(model.alphabet.symbols)
        symbols.sort()
        symbols = [model.terminal_symbol] + symbols   
        values = [OrderedDict(zip(symbols, model.last_token_probabilities(x, symbols))) for x in sequences]            
        data = dict(zip(sequences, values))
        joblib.dump(data, path)
        dataloader = PickleDataLoader(path)
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = self.comparator,
                                               max_seq_length=20,parallel_cache=True, cache_from_dataloader=dataloader)
        result = self.learner(self.probability_partitioner, pre_cache_queries_for_building_hipothesis = True).learn(teacher, verbose = True)        
        extracted_model = result.model     
        os.remove(path)
        self.assertTrue(self.comparator.are_equivalent(model, extracted_model))

    def test_tomitas_1_w_data_loader_cache_without_parallel_cache(self):
        model = WeightedTomitasGrammars.get_automaton_1()
        generator = UniformLengthSequenceGenerator(alphabet=model.alphabet, max_seq_length=5, min_seq_length=0)
        sequences = generator.generate_words(2000)
        path = "./test_dataset"
        symbols = list(model.alphabet.symbols)
        symbols.sort()
        symbols = [model.terminal_symbol] + symbols   
        values = [OrderedDict(zip(symbols, model.last_token_probabilities(x, symbols))) for x in sequences]            
        data = dict(zip(sequences, values))
        joblib.dump(data, path)
        dataloader = PickleDataLoader(path)
        teacher = PACBatchProbabilisticTeacher(model, 0.05, 0.01, comparator = self.comparator,
                                               max_seq_length=20,parallel_cache=False, cache_from_dataloader=dataloader)
        result = self.learner(self.probability_partitioner, pre_cache_queries_for_building_hipothesis = True).learn(teacher, verbose = True)        
        extracted_model = result.model     
        os.remove(path)
        self.assertTrue(self.comparator.are_equivalent(model, extracted_model))