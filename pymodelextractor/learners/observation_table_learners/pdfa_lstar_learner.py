import time

from pythautomata.automata.wheighted_automaton_definition.weighted_automaton import WeightedAutomaton
from pythautomata.base_types.sequence import Sequence
from pymodelextractor.learners.observation_table_learners.pdfa_observation_table import PDFAObservationTable, \
    epsilon
from pymodelextractor.learners.observation_table_learners.translators.pdfa_lstar_observation_table_translator import \
     PDFALStarObservationTableTranslation
from pymodelextractor.learners.pdfa_learner import PDFALearner
from pymodelextractor.teachers.probabilistic_teacher import ProbabilisticTeacher
from pymodelextractor.learners.learning_result import LearningResult

class PDFALStarLearner(PDFALearner):

    def __init__(self):
        self.model_translator = PDFALStarObservationTableTranslation()
        self.terminal_symbol = None
        self._teacher = None
        self.tolerance = None

    def learn(self, teacher: ProbabilisticTeacher, verbose: bool = False) -> LearningResult:
        self.terminal_symbol = teacher.terminal_symbol
        self._teacher = teacher
        self.tolerance = teacher.tolerance
        start_time = time.time()
        self.reset()
        if verbose: print("\n\n***** Learning started at:", start_time, "*****\n\n")

        model = None
        model_learned = False
        counter = 0
        counter_example_count = 0

        while not model_learned:
            if verbose: print("*** Start Iter", counter, "***")
            counter += 1
            start_inter_time = time.time()

            if verbose: print("Closing table...")
            self.__close()

            if verbose: print("Making table consistent...")
            self.__make_consistent()

            if verbose: print("Translating...")
            model = self.model_translator.translate(self.observation_table, self.tolerance, self.terminal_symbol)
            if verbose: print("Performing Equivalence Query...")
            model_learned, counterexample = self.perform_equivalence_query(model)
            if not model_learned:
                ce_time = time.time() - start_time
                if verbose: print("Found CounterExample after", ce_time, ":", counterexample)
                counter_example_count += 1
                self.__update_observation_table_with(counterexample)

            inter_time = time.time() - start_inter_time
            if verbose: print("*** Iter", counter, "finished after(secs):", inter_time, "- states:", len(model.weighted_states),
                  "- overall CE count:", counter_example_count, "***\n\n")

        if verbose: print("***** Learning completed successfully *****\n\n")

        info = {
            'equivalence_queries_count': self._teacher.equivalence_queries_count,
            'last_token_weight_queries_count': self._teacher.last_token_weight_queries_count,
            'observation_table':self.observation_table               
        }
        learningResult = LearningResult(model, len(model.weighted_states), info)
        return learningResult

    def reset(self):
        self.__build_observation_table()
        self.__initialize_observation_table()
        self._teacher.reset()

    def __build_observation_table(self):
        self.observation_table = PDFAObservationTable(self.__alphabet, self.tolerance)

    def __initialize_observation_table(self):
        self.observation_table.add_suffix(Sequence([self.terminal_symbol]))
        for s in self.__symbols:
            self.observation_table.add_suffix(Sequence([s]))
        self.__add_to_red(epsilon)
        for symbol in self.__symbols:
            seq = Sequence([symbol])
            self.__add_to_blue(seq)

    def __seq_prefix_weight(self, seq):
        return self._teacher.sequence_weight(seq)

    def __add_to_red(self, sequence: Sequence):
        if not self.observation_table.contains_in_red(sequence):
            self.observation_table.add_to_red(sequence)
            if not self.observation_table.contains_observation(sequence):
                self.observation_table[sequence] = self._get_filled_row_for(sequence)
            if self.observation_table.contains_in_blue(sequence):
                self.observation_table.remove_from_blue(sequence)

    def __add_to_blue(self, sequence: Sequence):
        if not self.observation_table.contains_in_blue(sequence) and \
                not self.observation_table.contains_in_red(sequence):
            weighted_sequence = (self.__seq_prefix_weight(sequence), sequence)
            self.observation_table.add_to_blue(weighted_sequence)
            if not self.observation_table.contains_observation(sequence):
                self.observation_table[sequence] = self._get_filled_row_for(sequence)

    def _get_filled_row_for(self, sequence: Sequence) -> list:
        required_suffixes = self.observation_table.get_suffixes()
        return self._teacher.last_token_weights(sequence, required_suffixes)

    def __close(self):
        violating_sequence = self.observation_table.get_violating_closedness_sequence()
        while violating_sequence is not None:
            self.__add_to_red(violating_sequence)
            for symbol in self.__symbols:
                new_blue_sequence = violating_sequence + symbol
                self.__add_to_blue(new_blue_sequence)
            violating_sequence = self.observation_table.get_violating_closedness_sequence()

    def __make_consistent(self):
        inconsistency = self.observation_table.find_inconsistency()
        while inconsistency is not None:
            self.__resolve_inconsistency(inconsistency)
            self.__close()
            inconsistency = self.observation_table.find_inconsistency()

    def __resolve_inconsistency(self, inconsistency):
        symbol = inconsistency.symbol
        different_suffix = inconsistency.different_suffix
        new_suffix = symbol + different_suffix
        self.observation_table.add_suffix(new_suffix)
        for sequence in self.observation_table.get_observed_sequences():
            self._fill_hole_for(sequence, new_suffix)

    def _fill_hole_for(self, sequence: Sequence, suffix):
        self.observation_table[sequence].append(self._teacher.last_token_weights(sequence, [suffix])[0])

    def __update_observation_table_with(self, counterexample):
        prefixes = counterexample.get_prefixes()
        for sequence in prefixes:
            self.__add_to_red(sequence)
        for prefix in self.observation_table.red:
            for symbol in self.__symbols:
                blue_prefix = prefix + symbol
                self.__add_to_blue(blue_prefix)

    # Helper methods

    @property
    def __alphabet(self):
        return self._teacher.alphabet

    @property
    def __symbols(self):
        return self._teacher.alphabet.symbols

    def perform_equivalence_query(self, model):
        return self._teacher.equivalence_query(model)
