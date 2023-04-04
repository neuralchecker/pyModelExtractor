import unittest

from pymodelextractor.teachers.general_teacher import \
    GeneralTeacher
from pymodelextractor.factories.lstar_factory import LStarFactory
from pythautomata.automata_definitions.tomitas_grammars import TomitasGrammars
from pythautomata.model_comparators.hopcroft_karp_comparison_strategy import \
    HopcroftKarpComparisonStrategy as ComparisonStrategy
from pythautomata.model_comparators.dfa_comparison_strategy import DFAComparisonStrategy
from pymodelextractor.learners.observation_table_learners.general_observation_table import GeneralObservationTable
from pythautomata.base_types.sequence import Sequence


class TestBaseObservationTable(unittest.TestCase):
    def get_observation_table(self, automaton) -> GeneralObservationTable:
        learner = LStarFactory.get_dfa_lstar_learner()

        result = learner.learn(GeneralTeacher(automaton, DFAComparisonStrategy()))
        return result.info['observation_table']
    
    def test_lstar_with_initialized_base_table(self):
        tomitas_automaton = TomitasGrammars.get_automaton_1()

        base_learner = LStarFactory.get_dfa_lstar_learner()
        base_learner._build_observation_table()
        base_learner._symbols = tomitas_automaton.alphabet.symbols
        base_learner._teacher = GeneralTeacher(tomitas_automaton, DFAComparisonStrategy())
        base_learner._initialize_observation_table()
        observation_table = base_learner._observation_table

        teacher = GeneralTeacher(tomitas_automaton, DFAComparisonStrategy())
        learner = LStarFactory.get_dfa_lstar_learner()

        result = learner.learn(teacher, observation_table)

        assert ComparisonStrategy().are_equivalent(result.model, tomitas_automaton)

    def test_lstar_with_completed_base_table(self):
        tomitas_automaton = TomitasGrammars.get_automaton_1()

        base_learner = LStarFactory.get_dfa_lstar_learner()
        base_teacher = GeneralTeacher(tomitas_automaton, DFAComparisonStrategy())
        base_result = base_learner.learn(base_teacher)
        observation_table = base_result.info['observation_table']
        
        teacher = GeneralTeacher(tomitas_automaton, DFAComparisonStrategy())
        learner = LStarFactory.get_dfa_lstar_learner()

        result = learner.learn(teacher, observation_table)

        assert ComparisonStrategy().are_equivalent(result.model, tomitas_automaton)

    def test_lstar_with_large_base_table(self):
        tomitas_automaton = TomitasGrammars.get_automaton_1()

        observation_table = self.generate_full_observation_table(column_limit=2, 
                                                                 row_limit=3, automaton=tomitas_automaton)
        
        teacher = GeneralTeacher(tomitas_automaton, DFAComparisonStrategy())
        learner = LStarFactory.get_dfa_lstar_learner()

        result = learner.learn(teacher, observation_table)

        assert ComparisonStrategy().are_equivalent(result.model, tomitas_automaton)

    def test_lstar_with_extra_large_base_table(self):
        tomitas_automaton = TomitasGrammars.get_automaton_1()

        observation_table = self.generate_full_observation_table(column_limit=3,
                                                                 row_limit=5, automaton=tomitas_automaton)
        
        teacher = GeneralTeacher(tomitas_automaton, DFAComparisonStrategy())
        learner = LStarFactory.get_dfa_lstar_learner()

        result = learner.learn(teacher, observation_table)

        assert ComparisonStrategy().are_equivalent(result.model, tomitas_automaton)

    def generate_full_observation_table(self, column_limit, row_limit, automaton):
        sequences = self.generate_all_seqeunces_with_limit(limit= row_limit, alphabet=automaton.alphabet)
        observation_table = GeneralObservationTable()
        exp_suffixes = self.generate_all_seqeunces_with_limit(limit= column_limit, alphabet=automaton.alphabet) 
        
        for sequence in exp_suffixes:
            observation_table.exp.append(sequence)

        for prefix in sequences:
            observation_table.add_to_blue(prefix)
            for suffix in exp_suffixes:
                sequence = prefix+suffix
                observation_table.__setitem__(sequence=sequence,
                                              observationsRow=[automaton.process_query(sequence)])
                
        return observation_table
        
    def generate_all_seqeunces_with_limit(self, limit, alphabet):
        sequences = [Sequence()]
        last_len_sequences = sequences
        for _ in range(limit):   
            new_sequences = []
            for sequence in last_len_sequences:
                for suffix in alphabet.symbols:
                    new_sequences.append(sequence.append(suffix.value))
            last_len_sequences = new_sequences
            sequences = sequences + new_sequences
                    
        return sequences
    
    def test_partitioned_lstar(self):
        automaton = TomitasGrammars.get_automaton_7()
        
        learner = LStarFactory.get_dfa_lstar_learner(max_query_length=4)

        teacher = GeneralTeacher(automaton, ComparisonStrategy())
        partial_result = learner.learn(teacher)

        partial_observation_table = partial_result.info['observation_table']

        learner = LStarFactory.get_dfa_lstar_learner()
        final_result = learner.learn(teacher, partial_observation_table)

        assert ComparisonStrategy().are_equivalent(final_result.model, automaton)

    def test_lstar_with_partial_red_table(self):
        automaton = TomitasGrammars.get_automaton_7()
        observation_table = self.get_observation_table(automaton)

        observation_table.red = set()

        learner = LStarFactory.get_partial_dfa_lstar_learner()
        
        result = learner.learn(GeneralTeacher(automaton, ComparisonStrategy()),
                               observation_table
                               )
        assert ComparisonStrategy().are_equivalent(result.model, automaton)

    def test_lstar_with_partial_blue_table(self):
        automaton = TomitasGrammars.get_automaton_7()
        observation_table = self.get_observation_table(automaton)

        observation_table.blue = set()

        learner = LStarFactory.get_partial_dfa_lstar_learner()
        
        result = learner.learn(GeneralTeacher(automaton, ComparisonStrategy()),
                               observation_table
                               )
        assert ComparisonStrategy().are_equivalent(result.model, automaton)
        
    def test_lstar_with_partial_obs_table(self):
        automaton = TomitasGrammars.get_automaton_7()
        observation_table = self.get_observation_table(automaton)

        observation_table.observations = dict()

        learner = LStarFactory.get_partial_dfa_lstar_learner()
        
        result = learner.learn(GeneralTeacher(automaton, ComparisonStrategy()),
                               observation_table
                               )
        assert ComparisonStrategy().are_equivalent(result.model, automaton)


        
    

    