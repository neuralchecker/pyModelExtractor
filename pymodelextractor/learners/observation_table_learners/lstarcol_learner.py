from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.alphabet import Alphabet
from pymodelextractor.teachers.teacher import Teacher
from pymodelextractor.learners.learner import Learner
from pymodelextractor.learners.observation_table_learners.observation_table import epsilon
from pymodelextractor.learners.observation_table_learners.translators.fa_observation_table_translator import FAObservationTableTranslator
from pymodelextractor.learners.learning_result import LearningResult
from pymodelextractor.learners.observation_table_learners.lstar_learner import LStarObservationTable

class LStarColLearner(Learner):

    def __init__(self):
        self._model_translator = FAObservationTableTranslator()

    def learn(self, teacher: Teacher) -> LearningResult:        
        self._teacher = teacher
        self._build_observation_table()
        self._initialize_observation_table()
        model = None
        answer = False
        counter = 1

        while not answer:
            self._close()
            model = self._model_translator.translate(self._observation_table, self._alphabet)
            answer, counterexample = self._teacher.equivalence_query(model)
            if not answer:            
                self._update_observation_table_with(counterexample)
            counter += 1
        return self._learning_results_for(model)

    def _build_observation_table(self):
        self._observation_table = LStarObservationTable(self._alphabet)

    def _initialize_observation_table(self):
        self._observation_table.exp = [epsilon]
        self._add_to_red(epsilon)
        for symbol in self._symbols:
            self._add_to_blue(Sequence(symbol))

    def _fill_hole_for(self, sequence: Sequence):
        suffix = self._observation_table.exp[-1]
        self._observation_table[sequence].append(self._teacher.membership_query(sequence + suffix))

    def _close(self):
        while True:
            blueSequence = self._get_closedness_violation_sequence()
            if blueSequence is None:
                return
            self._move_from_blue_to_red(blueSequence)
            for symbol in self._symbols:
                newBlueSequence = blueSequence + symbol
                self._add_to_blue(newBlueSequence)

    def _get_closedness_violation_sequence(self):
        return next(filter(self._no_same_row_exists_in_red, self._observation_table.blue), None)

    def _update_observation_table_with(self, counterexample):
        suffixes = counterexample.get_suffixes()
        for sequence in suffixes:
            self._add_to_suffixes(sequence)

    def _add_to_suffixes(self, sequence: Sequence):
        if sequence not in self._observation_table.exp:
            self._observation_table.exp.append(sequence)
            self._fill_last_column()

    def _fill_last_column(self) -> list:
        for sequence in self._observation_table.observations:
            self._fill_hole_for(sequence)

    def _add_to_red(self, sequence: Sequence):
        if sequence not in self._observation_table.red:
            self._observation_table.red.add(sequence)
            self._observation_table[sequence] = self._get_filled_row_for(sequence)

    def _add_to_blue(self, sequence: Sequence):
        if not sequence in self._observation_table.blue:
            self._observation_table.blue.add(sequence)
            self._observation_table[sequence] = self._get_filled_row_for(sequence)

    def _get_filled_row_for(self, sequence: Sequence) -> list:
        requiredSuffixes = self._observation_table.exp
        row = []
        for suffix in requiredSuffixes:
            result = self._teacher.membership_query(sequence + suffix)
            row.append(result)
        return row

    def _learning_results_for(self, model):
        numberOfStates = len(model.states) if model is not None else 0
        info = {
            'equivalence_queries_count': self._teacher.equivalence_queries_count,
            'membership_queries_count': self._teacher.membership_queries_count,
            'observation_table': self._observation_table
        }
        return LearningResult(model, numberOfStates, info)

    # Helper methods

    @property
    def _alphabet(self):
        return self._teacher.alphabet

    @property
    def _symbols(self):
        return self._teacher.alphabet.symbols

    def _no_same_row_exists_in_red(self, blueSequence: Sequence) -> bool:
        return not self._observation_table.same_row_exists_in_red(blueSequence)

    def _move_from_blue_to_red(self, blueSequence: Sequence):
        self._observation_table.move_from_blue_to_red(blueSequence)
