import time
import numpy as np

from pythautomata.base_types.sequence import Sequence
from pythautomata.model_comparators.wfa_comparison_strategy import WFAComparator
from pythautomata.model_comparators.wfa_tolerance_comparison_strategy import WFAToleranceComparator

from pymodelextractor.learners.observation_table_learners.pdfa_observation_table import PDFAObservationTable, \
    epsilon
from pymodelextractor.learners.observation_table_learners.translators.pdfa_lstarcol_observation_table_translator import \
    PDFALStarColObservationTableTranslator
from pymodelextractor.learners.observation_table_learners.translators.pdfa_observation_table_translator import \
    PDFAObservationTableTranslator
from pymodelextractor.teachers.probabilistic_teacher import ProbabilisticTeacher
from pymodelextractor.learners.learning_result import LearningResult


class PDFALStarColLearner:

    def __init__(self, comparator: WFAComparator = None, model_translator: PDFAObservationTableTranslator = None):
        self.terminal_symbol = None
        self._teacher = None
        if comparator is None:
            self.comparator = WFAToleranceComparator()
        else:
            self.comparator = comparator
        if model_translator is None:
            self.model_translator = PDFALStarColObservationTableTranslator()
        else:
            self.model_translator = model_translator

    def learn(self, teacher: ProbabilisticTeacher, verbose: bool = False) -> LearningResult:
        self.terminal_symbol = teacher.terminal_symbol
        self._teacher = teacher
        start_time = time.time()
        self.reset()
        if verbose: print("\n\n***** Learning started at:", start_time, "*****\n\n")

        model = None
        model_learned = False
        counter = 0
        counter_example_count = 0
        last_size = 0
        while not model_learned:
            if verbose: print("*** Start Iter", counter, "***")
            counter += 1
            start_inter_time = time.time()

            if verbose: print("Closing table...")
            self.__close()
            if verbose: print("Translating...")
            model = self.model_translator.translate(self.observation_table, self.terminal_symbol, self.comparator)
            size = len(model.weighted_states)
            assert size > last_size, 'Possible infinite loop'
            last_size = size
            if verbose: print("Performing Equivalence Query...")
            model_learned, counterexample = self.perform_equivalence_query(model)
            if not model_learned:
                ce_time = time.time() - start_time
                if verbose: print("Found CounterExample after", ce_time, ":", counterexample)
                counter_example_count += 1
                self.__update_observation_table_with(counterexample, model)

            inter_time = time.time() - start_inter_time
            if verbose: print("*** Iter", counter, "finished after(secs):", inter_time, "- states:",
                              len(model.weighted_states),
                              "- overall CE count:", counter_example_count, "***\n\n")

        if verbose: print("***** Learning completed successfully *****\n\n")       
        return self._learning_results_for(model)
    
    def _learning_results_for(self, model):
        info = {
            'equivalence_queries_count': self._teacher.equivalence_queries_count,
            'last_token_weight_queries_count': self._teacher.last_token_weight_queries_count,
            'observation_table': self.observation_table
        }
        learningResult = LearningResult(model, len(model.weighted_states), info)
        return learningResult

    def reset(self):
        self.__build_observation_table()
        self.__initialize_observation_table()
        self._teacher.reset()

    def __build_observation_table(self):
        self.observation_table = PDFAObservationTable(self.__alphabet, self.comparator)

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

    def _fill_hole_for(self, sequence: Sequence, suffixes):
        self.observation_table[sequence].extend(self._teacher.last_token_weights(sequence, suffixes))

    def __update_observation_table_with(self, counterexample, proposed_model):
        all_suffixes = []
        count, differing_symbol = self.__get_shortest_counterexample_with_symbol(counterexample, proposed_model)
        count = count + differing_symbol
        suffixes = count.get_suffixes()
        for suffix in suffixes:
            added = self.observation_table.add_suffix(suffix)
            if added:
                all_suffixes.append(suffix)
        for sequence in self.observation_table.get_observed_sequences():
            self._fill_hole_for(sequence, all_suffixes)

    def __get_shortest_counterexample_with_symbol(self, counterexample, proposed_model):
        symbols = list(self.__symbols)
        symbols.append(self.terminal_symbol)
        suffixes = counterexample.get_suffixes()
        suffixes.reverse()
        for suffix in suffixes:
            is_counterexample, symbol = self.__check_counterexample(suffix, symbols, proposed_model)
            if is_counterexample:
                return suffix, symbol
        return None, None

    def __check_counterexample(self, suffix, symbols, proposed_model):
        if isinstance(self.comparator, WFAToleranceComparator):
            model_values = np.array(proposed_model.last_token_probabilities(suffix, symbols))
            teacher_values = np.array(self._teacher.last_token_weights(suffix, symbols))
            diff = abs(model_values - teacher_values)
            if max(diff) > self.comparator.tolerance:
                return True, symbols[np.argmax(diff)]
            return False, None
        else:
            model_values = proposed_model.last_token_probabilities(suffix, symbols)
            teacher_values = self._teacher.last_token_weights(suffix, symbols)
            for i, value in enumerate(model_values):
                if not self.comparator.equivalent_values(value, teacher_values[i]):
                    return True, symbols[i]
            return False, None

    # Helper methods

    @property
    def __alphabet(self):
        return self._teacher.alphabet

    @property
    def __symbols(self):
        return self._teacher.alphabet.symbols

    def perform_equivalence_query(self, model):
        return self._teacher.equivalence_query(model)

    def _within_tolerance(self, value1, value2, tolerance):
        return abs(value1 - value2) <= tolerance
