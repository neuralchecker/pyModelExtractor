import unittest
from pythautomata.automata.wheighted_automaton_definition.probabilistic_deterministic_finite_automaton \
     import ProbabilisticDeterministicFiniteAutomaton
from pythautomata.automata.wheighted_automaton_definition.weighted_state import WeightedState

from pythautomata.automata_definitions.weighted_tomitas_grammars import WeightedTomitasGrammars

from pymodelextractor.learners.observation_tree_learners.pdfa_quantization_n_ary_tree_learner \
     import PDFAQuantizationNAryTreeLearner
from pythautomata.model_comparators.wfa_tolerance_comparison_strategy import WFAToleranceComparator
from pythautomata.model_comparators.wfa_quantization_comparison_strategy import WFAQuantizationComparator

from pymodelextractor.teachers.pdfa_teacher import PDFATeacher

from pythautomata.utilities import pdfa_generator
from pythautomata.utilities import nicaud_dfa_generator

from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.symbol import SymbolStr

binaryAlphabet = Alphabet(frozenset((SymbolStr('0'), SymbolStr('1'))))


class TestPDFAQuantizantionNAryTreeLearnerOnSimilarDistributions(unittest.TestCase):

    def setUp(self):
        self.learner = PDFAQuantizationNAryTreeLearner()

    def test_1(self):
        sizes = [2,3,4,5,6,7,8,9,10,11,12,13,50, 100, 200, 300]
        n=20
        counter = 0
        pdfas = []
        #pbar = tqdm(total=n*len(sizes))
        for size in sizes:
            counter = 0
            for i in range(n):
                dfa = nicaud_dfa_generator.generate_dfa(alphabet = binaryAlphabet, nominal_size= size, seed = counter)
                dfa.name = "random_PDFA_nominal_size_"+str(size)+"_"+str(counter)     
                pdfa = pdfa_generator.pdfa_from_dfa(dfa,distributions= 5, max_shift = 0)           
                pdfas.append(pdfa)
                #joblib.dump(pdfa, filename = path+dfa.name)
                counter += 1    
                #pbar.update(1) 
       #pbar.close() 
        tolerance = 0.001
        partitions = int(1/tolerance)        
        partition_comparator = WFAQuantizationComparator(partitions)
        algorithms = [('QuantNaryTreeLearner', PDFAQuantizationNAryTreeLearner, partition_comparator, partitions)]

        for (algorithm_name,algorithm, comparator, param) in algorithms:
            for pdfa in pdfas:                
                pdfa_teacher = PDFATeacher(pdfa, comparator)
                learner = algorithm()
                learner.learn(pdfa_teacher, param)   
                
                    
    