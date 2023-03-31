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
from pythautomata.base_types.sequence import Sequence

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
        new_automaton.export()
        automaton.export()

        
        assert not ComparisonStrategy().are_equivalent(new_automaton, automaton)
    

        
