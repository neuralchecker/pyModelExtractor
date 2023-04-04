import unittest

from pymodelextractor.teachers.general_teacher import \
    GeneralTeacher
from pymodelextractor.factories.lstar_factory import LStarFactory
from pythautomata.automata_definitions.tomitas_grammars import TomitasGrammars
from pythautomata.model_comparators.hopcroft_karp_comparison_strategy import \
    HopcroftKarpComparisonStrategy as ComparisonStrategy
from pymodelextractor.learners.observation_table_learners.translators\
    .partial_dfa_translator import PartialDFATranslator
from pythautomata.model_comparators.dfa_comparison_strategy import DFAComparisonStrategy
from pythautomata.utilities.simple_dfa_generator import generate_dfa
from pymodelextractor.learners.observation_table_learners.general_observation_table \
    import GeneralObservationTable
from pythautomata.automata.deterministic_finite_automaton import \
    DeterministicFiniteAutomaton as DFA
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.symbol import SymbolStr
from pythautomata.base_types.state import State
from pythautomata.model_comparators.dfa_comparison_strategy import \
      DFAComparisonStrategy as DFAComparator
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.symbol import SymbolStr
from pymodelextractor.utils.time_bound_utilities import is_unix_system

binaryAlphabet = Alphabet(frozenset((SymbolStr('0'), SymbolStr('1'))))
zero = binaryAlphabet['0']
one = binaryAlphabet['1']
a = SymbolStr('a')
b = SymbolStr('b')
abAlphabet = Alphabet(frozenset((a, b)))
epsilon = Sequence([])

class TestPartialDFATranslator(unittest.TestCase):
    def get_observation_table(self, automaton) -> GeneralObservationTable:
        learner = LStarFactory.get_dfa_lstar_learner()

        result = learner.learn(GeneralTeacher(automaton, DFAComparisonStrategy()))
        return result.info['observation_table']

    def test_with_complete_table(self):
        automaton = TomitasGrammars.get_automaton_1()
        
        new_automaton = PartialDFATranslator().translate(self.get_observation_table(automaton),
                                                         automaton.alphabet)
        
        assert ComparisonStrategy().are_equivalent(new_automaton, automaton)

    def test_with_big_table(self):
        alphabet = TomitasGrammars.get_automaton_1().alphabet
        automaton = generate_dfa(alphabet, seed=17)

        new_automaton = PartialDFATranslator().translate(self.get_observation_table(automaton),
                                                         automaton.alphabet)
        
        assert ComparisonStrategy().are_equivalent(new_automaton, automaton)

    def test_with_uncomplete_table(self):
        automaton = TomitasGrammars.get_automaton_7()
        observation_table = self.get_observation_table(automaton)
        observation_table.red.pop()
        observation_table.red.pop()
        observation_table.red.pop()


        new_automaton = PartialDFATranslator().translate(observation_table,
                                                         automaton.alphabet)
        
        assert not ComparisonStrategy().are_equivalent(new_automaton, automaton)

    def test_lstar_with_partial_translator(self):
        alphabet = TomitasGrammars.get_automaton_1().alphabet
        automaton = generate_dfa(alphabet, number_of_states=100, seed=17)
        learner = LStarFactory.get_partial_dfa_lstar_learner(max_query_length=5)

        partial_result = learner.learn(GeneralTeacher(automaton, DFAComparisonStrategy()))

        translated_automaton = PartialDFATranslator(). \
            translate(partial_result.info['observation_table'], automaton.alphabet)
            
        assert ComparisonStrategy().are_equivalent(partial_result.model, translated_automaton)

    def get_complex_alphabet(self):
        symbols = set()
        for symbol in range(100):
            symbols.add(SymbolStr(str(symbol)))

        return Alphabet(frozenset(symbols))

    def test_lstar_with_partial_translator_max_time(self):
        if is_unix_system():
            alphabet = self.get_complex_alphabet()
            automaton = generate_dfa(alphabet, number_of_states=50, seed=17)
            partial_learner = LStarFactory.get_partial_dfa_lstar_learner(max_time=3)

            teacher = GeneralTeacher(automaton, DFAComparisonStrategy())
            partial_result = partial_learner.learn(teacher)

            translated_automaton = PartialDFATranslator(). \
                translate(partial_result.info['observation_table'], automaton.alphabet)
                
            assert ComparisonStrategy().are_equivalent(partial_result.model, translated_automaton)

            learner = LStarFactory.get_partial_dfa_lstar_learner()

            result = learner.learn(teacher)

            assert ComparisonStrategy().are_equivalent(result.model, automaton)

    def test_lstar_with_partial_blue_table(self):
        automaton = TomitasGrammars.get_automaton_7()
        observation_table = self.get_observation_table(automaton)

        observation_table.blue = set()

        new_automaton = PartialDFATranslator().translate(observation_table,
                                                       automaton.alphabet)
        
        assert ComparisonStrategy().are_equivalent(new_automaton, automaton)
        
    def test_lstar_with_partial_obs_table(self):
        automaton = TomitasGrammars.get_automaton_7()
        observation_table = self.get_observation_table(automaton)

        observation_table.observations = dict()

        new_automaton = PartialDFATranslator().translate(observation_table,
                                                       automaton.alphabet)
        
        assert not ComparisonStrategy().are_equivalent(new_automaton, automaton)
    
    def test_partial_tomitas7_automaton(self):
        automaton = TomitasGrammars.get_automaton_7()
        
        stateQ1 = State("state0", True)
        stateQ1.add_transition(one, stateQ1)
        stateQ1.add_transition(zero, stateQ1)

        partial_expected_automaton = DFA(binaryAlphabet, stateQ1,
                                            set([stateQ1]), ComparisonStrategy(),
                                            "Tomita's grammar 7 automaton")
        
        learner = LStarFactory.get_partial_dfa_lstar_learner(max_query_length=2)

        partial_result = learner.learn(GeneralTeacher(automaton, DFAComparisonStrategy()))

        assert ComparisonStrategy().are_equivalent(partial_result.model, 
        partial_expected_automaton)
 
    def test_with_unclosed_table(self):
        state0 = State("State 0", True)
        state1 = State("State 1")

        state0.add_transition(a, state1)
        state0.add_transition(b, state1)
        state1.add_transition(a, state1)
        state1.add_transition(b, state1)

        comparator = DFAComparator()

        partial_expected_automaton = DFA(abAlphabet, state0,
                   set([state0, state1]), comparator, "Automaton with unclosed OT")
        
        obs_table = GeneralObservationTable()
        obs_table.red = set()
        obs_table.blue = set()
        obs_table.observations = {}

        obs_table.red.add(epsilon)
        obs_table.blue.add(Sequence([a]))
        obs_table.blue.add(Sequence([b]))

        obs_table[epsilon] = [True]
        obs_table[Sequence([a])] = [False]
        obs_table[Sequence([b])] = [False]

        translated_automaton = PartialDFATranslator().translate(obs_table,
                                                            partial_expected_automaton.alphabet)

        assert ComparisonStrategy().are_equivalent(translated_automaton, 
                                                   partial_expected_automaton)


    def create_automaton_1(self):
        state0 = State("State 0", True)
        state1 = State("State 1")

        state0.add_transition(a, state1)
        state0.add_transition(b, state1)
        state1.add_transition(a, state1)
        state1.add_transition(b, state0)

        comparator = DFAComparator()

        return DFA(abAlphabet, state0,
                   set([state0, state1]), comparator, "Automaton with inconsistent OT")

    def create_automaton_2(self):
        stateA = State("State A", True)
        stateB = State("State B")

        stateA.add_transition(a, stateB)
        stateA.add_transition(b, stateB)
        stateB.add_transition(a, stateB)
        stateB.add_transition(b, stateB)

        comparator = DFAComparator()

        return DFA(abAlphabet, stateA,
                   set([stateA, stateB]), comparator, "Automaton with inconsistent OT")

    def create_inconsistent_observation_table(self):
        obs_table = GeneralObservationTable()
        obs_table.red = set()
        obs_table.blue = set()
        obs_table.observations = {}

        obs_table.red.add(epsilon)
        obs_table.red.add(Sequence([a]))
        obs_table.red.add(Sequence([b]))
        obs_table.red.add(Sequence([b, b]))
        obs_table.blue.add(Sequence([a, a]))
        obs_table.blue.add(Sequence([a, b]))
        obs_table.blue.add(Sequence([b, a]))
        obs_table.blue.add(Sequence([b, b, a]))
        obs_table.blue.add(Sequence([b, b, b]))

        obs_table[epsilon] = [True]
        obs_table[Sequence([a])] = [False]
        obs_table[Sequence([b])] = [False]
        obs_table[Sequence([b, b])] = [False]
        obs_table[Sequence([a, a])] = [False]
        obs_table[Sequence([a, b])] = [True]
        obs_table[Sequence([b, a])] = [False]
        obs_table[Sequence([b, b, a])] = [False]
        obs_table[Sequence([b, b, b])] = [False]

        return obs_table
    
    def test_with_inconsistent_table(self):
        partial_expected_automaton1 = self.create_automaton_1()
        
        partial_expected_automaton2 = self.create_automaton_2()
        
        obs_table = self.create_inconsistent_observation_table()
        
        translated_automaton = PartialDFATranslator().translate(obs_table,
                                                            partial_expected_automaton1.alphabet)

        is_eq1 = ComparisonStrategy().are_equivalent(translated_automaton, 
                                                     partial_expected_automaton1)

        is_eq2 = ComparisonStrategy().are_equivalent(translated_automaton, 
                                                     partial_expected_automaton2)

        assert is_eq1 or is_eq2
        
    
        

