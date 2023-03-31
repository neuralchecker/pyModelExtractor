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


binaryAlphabet = Alphabet(frozenset((SymbolStr('0'), SymbolStr('1'))))
zero = binaryAlphabet['0']
one = binaryAlphabet['1']

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

        translated_automaton = PartialDFATranslator().translate(partial_result.info['observation_table'],
                                                            automaton.alphabet)
            
        assert ComparisonStrategy().are_equivalent(partial_result.model, translated_automaton)

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

        assert ComparisonStrategy().are_equivalent(partial_result.model, partial_expected_automaton)
        

