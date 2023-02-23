from pythautomata.base_types.sequence import Sequence
from pymodelextractor.learners.observation_table_learners.generic_observation_table import GenericObservationTable
from pymodelextractor.learners.learning_result import LearningResult
from pymodelextractor.learners.observation_table_learners.translators.fa_observation_table_translator import FAObservationTableTranslator
from pymodelextractor.teachers.generic_teacher import GenericTeacher
import time

lamda = Sequence()

no_verbose = 0
complete_verbose = 2

class GenericLStarLearner:
    def __init__(self, model_translator):
        self._model_translator = model_translator

    def _build_observation_table(self):
        self._observation_table = GenericObservationTable()
    
    def _initialize_observation_table(self):
        self._observation_table.exp = [lamda]
        self._add_to_red(lamda)
        for symbol in self._symbols:
            self._add_to_blue(Sequence((symbol,)))

    def _add_to_blue(self, sequence: Sequence):
        if not sequence in self._observation_table.blue:
            self._observation_table.blue.add(sequence)
            self._observation_table[sequence] = self._get_filled_row_for(
                sequence)

    def _add_to_red(self, sequence: Sequence):
        if sequence not in self._observation_table.red:
            redValue = self._get_filled_row_for(sequence)
            self._observation_table[sequence] = redValue
            self._observation_table.add_to_red(sequence, redValue)
    
    def _get_filled_row_for(self, sequence: Sequence) -> list:
        requiredSuffixes = self._observation_table.exp
        row = []
        for suffix in requiredSuffixes:
            result = self._teacher.membership_query(sequence + suffix)
            row.append(result)
        return row

    def learn(self, teacher: GenericTeacher, verbose: int = 0) -> LearningResult:
        start_time = time.time()
        teacher.verbose = verbose
        self.verbose = verbose
        if self.verbose != no_verbose:
            print("**** Started lstar learning ****")
        self._teacher = teacher
        self._symbols = self._teacher.alphabet.symbols
        self._build_observation_table()
        self._initialize_observation_table()
        model = None
        answer = False
        counter = 1
        counterexample_counter = 0

        while not answer:
            start_iteration_time = time.time()
            if self.verbose == complete_verbose:
                print(" # Starting iteration " + str(counter))
            self._close()
            self._make_consistent()
            
            self._model_translator._output_alphabet = self._teacher.output_alphabet
            model = self._model_translator.translate(
                self._observation_table, self._teacher.alphabet, self._teacher.output_alphabet)

            start_eq_time = time.time()
            answer, counterexample = self._teacher.equivalence_query(model)
            eq_duration = time.time() - start_eq_time

            if not answer:
                if self.verbose == complete_verbose:
                    print("    - Found counterexample in " + str(eq_duration) + "s -> " + str(counterexample))
                counterexample_counter += 1
                self._update_observation_table_with(counterexample)
            else:
                if self.verbose == complete_verbose:
                    print("    - Made equivalence query in " + str(eq_duration) + "s")

            duration = time.time() - start_iteration_time
            if self.verbose == complete_verbose:
                print("  # Iteration " + str(counter) + " ended, duration: " + str(duration) + "s")
            counter += 1


        result = self._learning_results_for(model, time.time() - start_time)
        duration = time.time() - start_time
        if (self.verbose != no_verbose):
            print("**** Learning finished in " + str(duration) + "s using " + str(counterexample_counter) \
                + " counterexamples & final model ended with " + str(result.state_count) + " states ****" + '\n')
        return result

    def _update_observation_table_with(self, counterexample):
        prefixes = counterexample.get_prefixes()
        for sequence in prefixes:
            self._add_to_red(sequence)
            for symbol in self._symbols:
                suffixedSequence = sequence + symbol
                if suffixedSequence not in prefixes:
                    self._add_to_blue(suffixedSequence)

    def _learning_results_for(self, model, duration):
        numberOfStates = len(model.states) if model is not None else 0
        info = {
            'equivalence_queries_count': self._teacher.equivalence_queries_count,
            'membership_queries_count': self._teacher.membership_queries_count,
            'observation_table': self._observation_table,
            'duration': duration,
        }
        return LearningResult(model, numberOfStates, info)

    def _close(self):
        start_closing_time = time.time()
        while True:
            closedCounterExample = self._observation_table.is_closed()
            if closedCounterExample == None:
                duration = time.time() - start_closing_time
                if self.verbose == complete_verbose:
                    print("    . Closed table in " + str(duration) + "s")
                return
            self._observation_table.move_from_blue_to_red(closedCounterExample)
            self._add_suffixes_to_blue(closedCounterExample)

    def _add_suffixes_to_blue(self, sequence: Sequence):
        for symbol in self._symbols:
            self._add_to_blue(sequence + symbol)

    def _make_consistent(self):
        while True:
            start_consistent_time = time.time()
            inconsistency = self._observation_table.find_inconsistency(self._teacher.alphabet)
            duration = time.time() - start_consistent_time
            if self.verbose == complete_verbose:
                print("    + Found Inconsistency in " + str(duration) + "s")
            if inconsistency == None:
                return
            
            start_consistent_time = time.time()
            self._resolve_inconsistency(inconsistency)
            duration = time.time() - start_consistent_time
            if self.verbose == complete_verbose:
                print("    + Resolved Inconsistency in " + str(duration) + "s")
            self._close()
    
    def _resolve_inconsistency(self, inconsistency):
        symbol = inconsistency.symbol + inconsistency.differenceSequence
        self._observation_table.exp.append(symbol)
        for sequence in self._observation_table.observations:
            self._fill_hole_for(sequence, symbol)
        self._observation_table.update_red_values()
        

    def _fill_hole_for(self, sequence: Sequence, suffix: Sequence):
        self._observation_table[sequence].append(
            self._teacher.membership_query(sequence + suffix))
    
    