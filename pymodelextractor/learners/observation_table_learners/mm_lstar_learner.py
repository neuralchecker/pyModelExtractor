from pymodelextractor.learners.learner import Learner
from pythautomata.base_types.sequence import Sequence
from pymodelextractor.learners.observation_table_learners.translators.mm_observation_table_translator import MMObservationTableTranslator
from pymodelextractor.learners.observation_table_learners.mm_observation_table import MMObservationTable
from pymodelextractor.learners.learning_result import LearningResult
from pymodelextractor.teachers.moore_machines_teacher import MooreMachineTeacher as MMTeacher
import time

lamda = Sequence()

class MMLStarLearner:
    def __init__(self):
        self._model_translator = MMObservationTableTranslator()

    def _build_observation_table(self):
        self._observation_table = MMObservationTable()
    
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

    def learn(self, teacher: MMTeacher, verbose: bool = False) -> LearningResult:
        start_time = time.time()
        teacher.verbose = verbose
        self.verbose = verbose
        if self.verbose:
            print("**** Started moore machines lstar ****")
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
            if self.verbose:
                print(" # Starting iteration " + str(counter))
            self._close()
            self._make_consistent()
            model = self._model_translator.translate(
                self._observation_table, self._teacher.alphabet, self._teacher.output_alphabet)
            answer, counterexample = self._teacher.equivalence_query(model, self.verbose)
            if not answer:
                counterexample_counter += 1
                self._update_observation_table_with(counterexample)
            duration = time.time() - start_iteration_time
            if self.verbose:
                print("  # Iteration " + str(counter) + " ended, duration: " + str(duration) + "s")
            counter += 1


        result = self._learning_results_for(model, time.time() - start_time)
        duration = time.time() - start_time
        if (self.verbose):
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
                if self.verbose:
                    print("    . Closed table in " + str(duration) + "s")
                return
            self._observation_table.move_from_blue_to_red(closedCounterExample)
            self._add_suffixes_to_blue(closedCounterExample)

    def _add_suffixes_to_blue(self, sequence: Sequence):
        for symbol in self._symbols:
            self._add_to_blue(sequence + symbol)

    def _make_consistent(self):
        start_consistent_time = time.time()
        while True:
            inconsistency = self._observation_table.find_inconsistency(self._teacher.alphabet)
            if inconsistency == None:
                duration = time.time() - start_consistent_time
                if self.verbose:
                    print("    + Made table consistent in " + str(duration) + "s")
                return

            self._resolve_inconsistency(inconsistency)
            self._close()
    
    def _resolve_inconsistency(self, inconsistency):
        symbol = inconsistency.symbol + inconsistency.differenceSequence
        self._observation_table.exp.append(symbol)
        for sequence in self._observation_table.observations:
            self._fill_hole_for(sequence, symbol)

    def _fill_hole_for(self, sequence: Sequence, suffix: Sequence):
        self._observation_table[sequence].append(
            self._teacher.membership_query(sequence + suffix))
    
    