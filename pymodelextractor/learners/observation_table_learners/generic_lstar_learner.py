from ast import Tuple
from pythautomata.base_types.sequence import Sequence
from pymodelextractor.learners.observation_table_learners.generic_observation_table import GenericObservationTable
from pymodelextractor.learners.learning_result import LearningResult
from pymodelextractor.teachers.generic_teacher import GenericTeacher
import time

lamda = Sequence()

no_log = 'none'
info_log = 'info'
debug_log = 'debug'
trace_log = 'trace'

class GenericLStarLearner:
    def __init__(self, model_translator, max_states = None, max_query_lenght = None):
        self._model_translator = model_translator
        self._max_states = max_states
        self._max_query_length = max_query_lenght
        self._history = []

    def _build_observation_table(self):
        self._observation_table = GenericObservationTable()
    
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
            
        

    def _add_to_red(self, sequence: Sequence) -> bool:
        if sequence not in self._observation_table.red:
            redValue, surpassed_max_query_len = self._get_filled_row_for(sequence)
            self._observation_table[sequence] = redValue
            self._observation_table.add_to_red(sequence, redValue)

            return surpassed_max_query_len
    
    def _get_filled_row_for(self, sequence: Sequence):
        required_suffixes = self._observation_table.exp
        row = []

        if (self._max_query_length is not None) and ( max(len(suffix) for suffix in required_suffixes) + len(sequence) > self._max_query_length):
            return row, True
        
        for suffix in required_suffixes:
            result = self._teacher.membership_query(sequence + suffix)
            row.append(result)
        return row, False

    def learn(self, teacher: GenericTeacher, log_hierachy: str = 'none') -> LearningResult:
        start_time = time.time()
        teacher.log_hierachy = log_hierachy
        self.log_hierachy = log_hierachy
        if self.log_hierachy != no_log:
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
            if self.log_hierachy == debug_log or self.log_hierachy == trace_log:
                print(" # Starting iteration " + str(counter))
            surpassed_max_query_len = self._close()
            
            if surpassed_max_query_len:
                return self._learning_results_for(self._history[-1] if len(self._history) > 0 else None, time.time() - start_time)

            surpassed_max_query_len = self._make_consistent()
            
            if surpassed_max_query_len:
                return self._learning_results_for(self._history[-1] if len(self._history) > 0 else None, time.time() - start_time)

            self._model_translator._output_alphabet = self._teacher.output_alphabet
            model = self._model_translator.translate(
                self._observation_table, self._teacher.alphabet, self._teacher.output_alphabet)
            self._history.append(model)

            start_eq_time = time.time()
            answer, counterexample = self._teacher.equivalence_query(model)
            eq_duration = time.time() - start_eq_time

            if (self._max_states is not None) and (len(model.states) > self._max_states):
                return self._learning_results_for(model, time.time() - start_time)

            if not answer:
                if self.log_hierachy == debug_log or self.log_hierachy == trace_log:
                    print("    - Found counterexample in " + str(eq_duration) + "s -> " + str(counterexample))
                counterexample_counter += 1

                surpassed_max_query_len = self._update_observation_table_with(counterexample)
                if surpassed_max_query_len:
                    return self._learning_results_for(self._history[-1] if len(self._history) > 0 else None, time.time() - start_time)
                
            else:
                if self.log_hierachy == debug_log or self.log_hierachy == trace_log:
                    print("    - Made equivalence query in " + str(eq_duration) + "s")

            duration = time.time() - start_iteration_time
            if self.log_hierachy == debug_log or self.log_hierachy == trace_log:
                print("  # Iteration " + str(counter) + " ended, duration: " + str(duration) + "s")
            counter += 1


        result = self._learning_results_for(model, time.time() - start_time)
        duration = time.time() - start_time
        if self.log_hierachy != no_log:
            print("**** Learning finished in " + str(duration) + "s using " + str(counterexample_counter) \
                + " counterexamples & final model ended with " + str(result.state_count) + " states ****" + '\n')
        return result

    def _update_observation_table_with(self, counterexample) -> bool:
        prefixes = counterexample.get_prefixes()
        for sequence in prefixes:
            surpassed_max_query_len = self._add_to_red(sequence)
            if surpassed_max_query_len:
                        return surpassed_max_query_len
            
            for symbol in self._symbols:
                suffixedSequence = sequence + symbol
                if suffixedSequence not in prefixes:
                    surpassed_max_query_len = self._add_to_blue(suffixedSequence)

                    if surpassed_max_query_len:
                        return surpassed_max_query_len
        return False
    
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
                if self.log_hierachy == trace_log:
                    print("    . Closed table in " + str(duration) + "s")
                return False
            self._observation_table.move_from_blue_to_red(closed_counter_example)
            surpassed_max_query_len = self._add_suffixes_to_blue(closed_counter_example)

            if surpassed_max_query_len:
                return surpassed_max_query_len
    

    def _add_suffixes_to_blue(self, sequence: Sequence) -> bool:
        for symbol in self._symbols:
            surpassed_max_query_len = self._add_to_blue(sequence + symbol)
            if surpassed_max_query_len:
                return surpassed_max_query_len
            
        return False

    def _make_consistent(self) -> bool:
        while True:
            start_consistent_time = time.time()
            inconsistency = self._observation_table.find_inconsistency(self._teacher.alphabet)
            duration = time.time() - start_consistent_time
            if self.log_hierachy == trace_log:
                print("    + Found Inconsistency in " + str(duration) + "s")
            if inconsistency == None:
                return False
            
            start_consistent_time = time.time()
            surpassed_max_query_len = self._resolve_inconsistency(inconsistency)

            if surpassed_max_query_len:
                return surpassed_max_query_len

            duration = time.time() - start_consistent_time
            if self.log_hierachy == trace_log:
                print("    + Resolved Inconsistency in " + str(duration) + "s")
            
            surpassed_max_query_len = self._close()

            if surpassed_max_query_len:
                return surpassed_max_query_len
    
    def _resolve_inconsistency(self, inconsistency):
        symbol = inconsistency.symbol + inconsistency.differenceSequence
        self._observation_table.exp.append(symbol)
        for sequence in self._observation_table.observations:
            surpassed_max_query_len = self._fill_hole_for(sequence, symbol)
            if surpassed_max_query_len:
                return surpassed_max_query_len

        self._observation_table.update_red_values()
        return False
        

    def _fill_hole_for(self, sequence: Sequence, suffix: Sequence):
        if (self._max_query_length is not None) and (len(sequence) + len(suffix) > self._max_query_length):
            return True
        
        self._observation_table[sequence].append(
            self._teacher.membership_query(sequence + suffix))
        
        return False
    
    