import unittest

from pymodelextractor.learners.observation_table_learners.generic_lstar_learner import \
    GenericLStarLearner
from pymodelextractor.learners.observation_table_learners.lstar_learner import \
    LStarLearner
from pymodelextractor.teachers.automaton_teacher import \
    DeterministicFiniteAutomatonTeacher as AutomatonTeacher
from pymodelextractor.learners.observation_table_learners.mm_lstar_learner import \
    MMLStarLearner
from pymodelextractor.teachers.generic_teacher import \
    GenericTeacher
from pymodelextractor.teachers.moore_machines_teacher import \
    MooreMachineTeacher
from pymodelextractor.learners.observation_table_learners.lstar_factory import LStarFactory
from pymodelextractor.teachers.automaton_teacher import \
    DeterministicFiniteAutomatonTeacher as DFATeacher
from pythautomata.automata.deterministic_finite_automaton import \
    DeterministicFiniteAutomaton
from pythautomata.automata_definitions.tomitas_grammars import TomitasGrammars
from pythautomata.automata_definitions.bollig_habermehl_kern_leucker_automata import BolligHabermehlKernLeuckerAutomata
from pythautomata.automata_definitions.omlin_giles_automata import OmlinGilesAutomata
from pythautomata.model_comparators.hopcroft_karp_comparison_strategy import \
    HopcroftKarpComparisonStrategy as ComparisonStrategy
from pythautomata.model_comparators.moore_machine_comparison_strategy import MooreMachineComparisonStrategy
from pythautomata.model_comparators.dfa_comparison_strategy import DFAComparisonStrategy
from itertools import chain
import time
from pythautomata.utilities.automata_converter import AutomataConverter
from pythautomata.utilities.simple_dfa_generator import generate_dfa



class TestLStarLearner(unittest.TestCase):
    def setUp(self):
        self.learner = LStarLearner()

    def teacher(self, automaton: DeterministicFiniteAutomaton):
        return AutomatonTeacher(automaton, ComparisonStrategy())

    def test_tomitas_1(self):
        grammar1 = TomitasGrammars.get_automaton_1()
        teacher = self.teacher(grammar1)
        result = self.learner.learn(teacher)
        assert ComparisonStrategy().are_equivalent(
            result.model, grammar1)

    def test_tomitas_2(self):
        grammar2 = TomitasGrammars.get_automaton_2()
        teacher = self.teacher(grammar2)
        result = self.learner.learn(teacher)
        assert ComparisonStrategy().are_equivalent(
            result.model, grammar2)

    def test_tomitas_3(self):
        grammar3 = TomitasGrammars.get_automaton_3()
        teacher = self.teacher(grammar3)
        result = self.learner.learn(teacher)
        assert ComparisonStrategy().are_equivalent(
            result.model, grammar3)

    def test_tomitas_4(self):
        grammar4 = TomitasGrammars.get_automaton_4()
        teacher = self.teacher(grammar4)
        result = self.learner.learn(teacher)
        assert ComparisonStrategy().are_equivalent(
            result.model, grammar4)

    def test_against_many_DFAs(self):
        mergedAutomata = list(chain(TomitasGrammars.get_all_automata(),
                                    BolligHabermehlKernLeuckerAutomata.get_all_automata(),
                                    OmlinGilesAutomata.get_all_automata()))
        for automaton in mergedAutomata:
            teacher = self.teacher(automaton)
            result = self.learner.learn(teacher)
            assert ComparisonStrategy().are_equivalent(
                result.model, automaton)
       
    def test_generic_learner(self):
        alphabet = TomitasGrammars.get_automaton_7()._alphabet
        iters = 10
        print("Benchmark for lstar algorithms started")
        for i in range(2):
            t = {'moore':[], 'dfa':[], 'generic':[], 'genericM':[]}
            num = (i+1)*100
            print(" - " + str(num)+" states, iteration n• " +  str(i+1))
            for _ in range(iters):
                automaton = generate_dfa(alphabet, num)
                moore = AutomataConverter().convert_dfa_to_moore_machine(automaton)
                pt, result = self.learnMoore(moore)
                assert MooreMachineComparisonStrategy().are_equivalent(result.model, moore)
                t['moore'].append(pt)

                pt, result = self.learnDfa(automaton)
                assert ComparisonStrategy().are_equivalent(result.model, automaton)
                t['dfa'].append(pt)


                pt, result = self.learnGeneric(LStarFactory.get_dfa_lstar_learner(), GenericTeacher(automaton, DFAComparisonStrategy()))
                assert ComparisonStrategy().are_equivalent(result.model, automaton)
                t['generic'].append(pt)

                pt, result = self.learnGeneric(LStarFactory.get_moore_machine_lstar_learner(), GenericTeacher(moore, MooreMachineComparisonStrategy()))
                assert MooreMachineComparisonStrategy().are_equivalent(result.model, moore)
                t['genericM'].append(pt)
                
            print("     + Moore avg time with " + str(iters) + " iterarions -> " + str(sum(t['moore']) / len(t['moore'])) +"s")
            print("     ^ Dfa avg time with " + str(iters) + " iterarions -> " + str(sum(t['dfa']) / len(t['dfa'])) +"s")
            print("     * Generic using dfa avg time with " + str(iters) + " iterarions -> " + str(sum(t['generic']) / len(t['generic'])) +"s")
            print("     * Generic using moore avg time with " + str(iters) + " iterarions -> " + str(sum(t['genericM']) / len(t['genericM'])) +"s")

    def learnMoore(self, moore):
        start = time.time()
        teacher = MooreMachineTeacher(moore)
        result = MMLStarLearner().learn(teacher)
        end_time = time.time() - start
        return end_time, result

    def learnDfa(self, automaton):
        start = time.time()
        teacher = DFATeacher(automaton, ComparisonStrategy())
        result = LStarLearner().learn(teacher)
        end_time = time.time() - start
        return end_time, result

    def learnGeneric(self, learner, teacher):
        start = time.time()
        result = learner.learn(teacher)
        end_time = time.time() - start
        return end_time, result
        