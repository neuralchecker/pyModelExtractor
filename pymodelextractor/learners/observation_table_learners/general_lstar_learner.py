from pythautomata.base_types.sequence import Sequence
from pymodelextractor.learners.observation_table_learners.general_observation_table\
      import GeneralObservationTable
from pymodelextractor.learners.learning_result import LearningResult
from pymodelextractor.teachers.general_teacher import GeneralTeacher
import time
from pymodelextractor.utils.time_bound_utilities import timeout
from pymodelextractor.learners.observation_table_learners.translators.partial_dfa_translator \
    import PartialDFATranslator

lamda = Sequence()

no_log = 0
info_log = 1
debug_log = 2
trace_log = 3

class GeneralLStarLearner:
    def __init__(self, model_translator, max_states = -1, max_query_lenght = -1, max_time = -1):
        self._model_translator = model_translator
        self._max_states = max_states
        self._max_query_length = max_query_lenght
        self._max_time = max_time
        self._last_model = None

    def _build_observation_table(self):
        self._observation_table = GeneralObservationTable()
    
    def _initialize_observation_table(self):
        self._observation_table.exp = [lamda]
        self._add_to_red(lamda)
        for symbol in self._symbols:
            self._add_to_blue(Sequence((symbol,)))

    def _add_to_blue(self, sequence: Sequence) -> bool:
        if not sequence in self._observation_table.blue:
            self._observation_table.blue.add(sequence)
            self._observation_table[sequence], surpassed_max_query_len = self._get_filled_row_for(
                sequence)
            
            return surpassed_max_query_len
        return False
            
    def _add_to_red(self, sequence: Sequence) -> bool:
        surpassed_len = False
        if sequence not in self._observation_table.red:
            redValue, surpassed_max_query_len = self._get_filled_row_for(sequence)
            self._observation_table[sequence] = redValue
            self._observation_table.add_to_red(sequence, redValue)

            if surpassed_max_query_len:
                surpassed_len = True
        return surpassed_len
    
    def _get_filled_row_for(self, sequence: Sequence):
        required_suffixes = self._observation_table.exp
        row = []
        
        for suffix in required_suffixes:
            result = self._teacher.membership_query(sequence + suffix)
            row.append(result)

        return row, self._surpassed_max_query_len(sequence, required_suffixes)
    
    def _surpassed_max_query_len(self, sequence, required_suffixes):
        return (self._max_query_length != -1) and \
            (max(len(suffix) for suffix in required_suffixes) + len(sequence) > self._max_query_length)
    
    def learn(self, teacher, observation_table: GeneralObservationTable = None,
               log_hierachy: int = 0) -> LearningResult:
        if self._max_time == -1:
            stopped_by_bounds, _ = self._learn(teacher, observation_table, log_hierachy)
            if stopped_by_bounds and type(self._model_translator) == PartialDFATranslator:
                self._last_model = self._model_translator.translate(self._observation_table,
                                                                self._teacher.alphabet,
                                                                self._teacher.output_alphabet)
            return self._learning_results_for(self._last_model, self._max_time)

        try:
            with timeout(self._max_time):
                stopped_by_bounds, _ = self._learn(teacher, observation_table, log_hierachy) 
                if stopped_by_bounds and type(self._model_translator) == PartialDFATranslator:
                    self._last_model = self._model_translator.translate(self._observation_table,
                                                                    self._teacher.alphabet,
                                                                    self._teacher.output_alphabet)
                return self._learning_results_for(self._last_model, self._max_time)
        except TimeoutError:
            if type(self._model_translator) == PartialDFATranslator:
                self._last_model = self._model_translator.translate(self._observation_table, 
                                                                    self._teacher.alphabet, 
                                                                    self._teacher.output_alphabet)
            return self._learning_results_for(self._last_model, self._max_time)

    def _learn(self, teacher: GeneralTeacher, observation_table: GeneralObservationTable = None,
                log_hierachy: int = 0) -> tuple[bool, LearningResult]:
        start_time = time.time()
        self.log_hierachy = log_hierachy
        if self.log_hierachy > no_log:
            print("**** Started lstar learning ****")
        self._teacher = teacher
        self._symbols = self._teacher.alphabet.symbols
        if observation_table is None:
            self._build_observation_table()
            self._initialize_observation_table()
        else:
            self._observation_table = observation_table
            self._observation_table.fill_observations(teacher)
        
        model = None
        answer = False
        counter = 1
        counterexample_counter = 0

        while not answer:
            start_iteration_time = time.time()
            if self.log_hierachy >= debug_log:
                print(" # Starting iteration " + str(counter))
            surpassed_max_query_len = self._close()
            
            if surpassed_max_query_len:
                return True, self._learning_results_for(self._last_model, time.time() - start_time)

            surpassed_max_query_len = self._make_consistent()
            
            if surpassed_max_query_len:
                return True, self._learning_results_for(self._last_model, time.time() - start_time)

            self._model_translator._output_alphabet = self._teacher.output_alphabet
            model = self._model_translator.translate(
                self._observation_table, self._teacher.alphabet, self._teacher.output_alphabet)
            self._last_model = model

            start_eq_time = time.time()
            answer, counterexample = self._teacher.equivalence_query(model)
            eq_duration = time.time() - start_eq_time

            if (self._max_states != -1) and (len(model.states) > self._max_states):
                return True, self._learning_results_for(model, time.time() - start_time)

            if not answer:
                if self.log_hierachy >= debug_log:
                    print("    - Found counterexample in " + str(eq_duration) + "s -> " + str(counterexample))
                counterexample_counter += 1

                surpassed_max_query_len = self._update_observation_table_with(counterexample)
                if surpassed_max_query_len:
                    return True, self._learning_results_for(self._last_model, time.time() - start_time)
                
            else:
                if self.log_hierachy >= debug_log:
                    print("    - Made equivalence query in " + str(eq_duration) + "s")

            duration = time.time() - start_iteration_time
            if self.log_hierachy >= debug_log:
                print("  # Iteration " + str(counter) + " ended, duration: " + str(duration) + "s")
            counter += 1

        result = self._learning_results_for(model, time.time() - start_time)
        duration = time.time() - start_time
        if self.log_hierachy > no_log:
            print("**** Learning finished in " + str(duration) + "s using " + str(counterexample_counter) \
                + " counterexamples & final model ended with " + str(result.state_count) + " states ****" + '\n')

        return False, result

    def _update_observation_table_with(self, counterexample) -> bool:
        prefixes = counterexample.get_prefixes()
        surpassed_max_query_len = False
        for sequence in prefixes:
            surpassed_max_query_len = self._add_to_red(sequence)
            
            for symbol in self._symbols:
                suffixedSequence = sequence + symbol
                if suffixedSequence not in prefixes:
                    if self._add_to_blue(suffixedSequence):
                        surpassed_max_query_len = True
        
        return surpassed_max_query_len
    
    def _learning_results_for(self, model, duration):
        number_of_states = len(model.states) if model is not None else 0
        info = {
            'equivalence_queries_count': self._teacher.equivalence_queries_count,
            'membership_queries_count': self._teacher.membership_queries_count,
            'observation_table': self._observation_table,
            'duration': duration,
        }
        return LearningResult(model, number_of_states, info)

    def _close(self) -> bool:
        start_closing_time = time.time()
        while True:
            closed_counter_example = self._observation_table.is_closed()
            if closed_counter_example == None:
                duration = time.time() - start_closing_time
                if self.log_hierachy >= trace_log:
                    print("    . Closed table in " + str(duration) + "s")
                return False
            self._observation_table.move_from_blue_to_red(closed_counter_example)
            surpassed_max_query_len = self._add_suffixes_to_blue(closed_counter_example)

            if surpassed_max_query_len:
                return surpassed_max_query_len
    

    def _add_suffixes_to_blue(self, sequence: Sequence) -> bool:
        surpassed_max_query_len = False
        for symbol in self._symbols:
            if self._add_to_blue(sequence + symbol):
                surpassed_max_query_len = True
            
        return surpassed_max_query_len

    def _make_consistent(self) -> bool:
        while True:
            start_consistent_time = time.time()
            inconsistency = self._observation_table.find_inconsistency(self._teacher.alphabet)
            duration = time.time() - start_consistent_time
            if self.log_hierachy >= trace_log:
                print("    + Found Inconsistency in " + str(duration) + "s")
            if inconsistency == None:
                return False
            
            start_consistent_time = time.time()
            surpassed_max_query_len = self._resolve_inconsistency(inconsistency)

            if surpassed_max_query_len:
                return surpassed_max_query_len

            duration = time.time() - start_consistent_time
            if self.log_hierachy >= trace_log:
                print("    + Resolved Inconsistency in " + str(duration) + "s")
            
            surpassed_max_query_len = self._close()

            if surpassed_max_query_len:
                return surpassed_max_query_len
    
    def _resolve_inconsistency(self, inconsistency):
        symbol = inconsistency.symbol + inconsistency.differenceSequence
        surpassed_max_query_len = False
        self._observation_table.exp.append(symbol)
        for sequence in self._observation_table.observations:
            if self._fill_hole_for(sequence, symbol):
                surpassed_max_query_len = True
            
        self._observation_table.update_red_values()
        return surpassed_max_query_len
        

    def _fill_hole_for(self, sequence: Sequence, suffix: Sequence):
        self._observation_table[sequence].append(
            self._teacher.membership_query(sequence + suffix))
        
        return self._surpassed_max_query_len(sequence, [suffix])
    
    