from typing import Union

from pymodelextractor.learners.learner import Learner
from pymodelextractor.learners.learning_result import LearningResult
from pymodelextractor.learners.observation_table_learners.observation_table import (
    epsilon, ObservationTable, TableInconsistency)
from pymodelextractor.learners.observation_table_learners.translators.fa_observation_table_translator import \
    FAObservationTableTranslator
from pymodelextractor.teachers.teacher import Teacher
from pythautomata.automata.deterministic_finite_automaton import \
    DeterministicFiniteAutomaton as DFA
from pythautomata.automata.symbolic_finite_automaton import \
    SymbolicFiniteAutomaton as SFA
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.symbol import Symbol
from pythautomata.boolean_algebra_learner.boolean_algebra_learner import \
    BooleanAlgebraLearner
from pythautomata.boolean_algebra_learner.closed_discrete_interval_learner import \
    ClosedDiscreteIntervalLearner as IntervalLearner
from pythautomata.utilities.automata_converter import AutomataConverter


class LambdaStarLearner(Learner):
    # TODO this should probably be instantiated in learn
    _observed_symbols: set[Symbol] = set()
    _o_t_translator = FAObservationTableTranslator()
    _algebra_learner: BooleanAlgebraLearner

    def __init__(self, boolean_algebra_learner: BooleanAlgebraLearner = IntervalLearner):
        self._algebra_learner = boolean_algebra_learner

    def learn(self, teacher: Teacher) -> LearningResult:
        answer: bool
        counter_example: Sequence
        observation_table: _ObservationTable = self._build_observation_table()
        self._initialize_observation_table(observation_table, teacher)
        model = self._build_model(observation_table)
        answer, counter_example = teacher.equivalence_query(model)
        self._update_with_new_counterexample(
            observation_table, teacher, counter_example)
        self._make_consistent(observation_table, teacher)

        while not answer:
            model = self._build_model(observation_table)
            answer, counter_example = teacher.equivalence_query(model)
            if not answer:
                self._update_with_new_counterexample(
                    observation_table, teacher, counter_example)

                self._close_table(observation_table, teacher)

                self._make_consistent(observation_table, teacher)

        # TODO add states counter
        return LearningResult(model, 0)

    def _update_with_new_counterexample(self, observation_table: '_ObservationTable', teacher: Teacher,
                                        counter_example: Sequence) -> None:

        self._update_observed_symbols(counter_example)
        self._update_observation_table_with_counterexample(
            observation_table, teacher, counter_example)

    def _make_consistent(self, observation_table: '_ObservationTable', teacher: Teacher):
        alphabet = Alphabet(frozenset(self._observed_symbols))
        while True:
            inconsistency = observation_table.find_inconsistency(alphabet)
            if inconsistency is None:
                return
            self._resolve_inconsistency(observation_table)
            self._close_table(observation_table, teacher)

    def _resolve_inconsistency(self, observation_table: '_ObservationTable'):

        pass

    def _build_model(self, observation_table: '_ObservationTable') -> SFA:
        # not truly a dfa as it might be missing transitions, but using a dfa with missing transitions is what we need
        evidence_automaton: DFA = self._o_t_translator.translate(
            observation_table, Alphabet(frozenset(self._observed_symbols)))
        return AutomataConverter.convert_dfa_to_sfa(evidence_automaton, self._algebra_learner)

    def _close_table(self, observation_table: '_ObservationTable', teacher: Teacher) -> None:
        while True:
            blue_sequence = self._get_closedness_violation_sequence(
                observation_table)
            if blue_sequence is None:
                return
            observation_table.move_from_blue_to_red(blue_sequence)
            for symbol in self._observed_symbols:
                new_blue_sequence = blue_sequence + symbol
                self._add_to_blue(observation_table,
                                  teacher, new_blue_sequence)

    def _get_closedness_violation_sequence(self, observation_table: '_ObservationTable') -> Union[Sequence, None]:
        return next(filter(observation_table.same_row_exists_in_red, observation_table.blue), None)

    def _update_observation_table_with_counterexample(self,
                                                      observation_table: '_ObservationTable', teacher: Teacher, counter_example: Sequence) -> None:

        # save it inside a set, if sequence is long enough, this will optimize the algorithm
        prefixes = set(counter_example.get_prefixes())

        for sequence in prefixes:
            self._add_to_red(observation_table, teacher, sequence)
            for symbol in self._observed_symbols:
                suffixed_sequence = sequence + symbol
                if suffixed_sequence not in prefixes:
                    self._add_to_blue(observation_table,
                                      teacher, suffixed_sequence)

    def _update_observed_symbols(self, sequence: Sequence) -> None:

        for symbol in sequence:
            self._observed_symbols.add(symbol)

    def _build_observation_table(self) -> '_ObservationTable':
        return _ObservationTable()

    def _initialize_observation_table(self, observation_table: '_ObservationTable', teacher: Teacher) -> None:
        observation_table.exp = [epsilon]
        self._add_to_red(observation_table, teacher, epsilon)
        # should never enter this for, leaving it here just in case
        for symbol in self._observed_symbols:
            assert(False)
            self._add_to_blue(observation_table, teacher, Sequence(symbol))

    def _add_to_red(self, observation_table: '_ObservationTable', teacher: Teacher, sequence: Sequence) -> None:
        if sequence not in observation_table.red:
            observation_table.red.add(sequence)
            observation_table[sequence] = self._get_filled_row_for(
                observation_table, teacher, sequence)

    def _add_to_blue(self, observation_table: '_ObservationTable', teacher: Teacher, sequence: Sequence) -> None:
        if not sequence in observation_table.blue:
            observation_table.blue.add(sequence)
            observation_table[sequence] = self._get_filled_row_for(
                observation_table, teacher, sequence)

    def _get_filled_row_for(self, observation_table: '_ObservationTable', teacher: Teacher, sequence: Sequence) -> list[bool]:
        row: list[bool] = []
        for suffix in observation_table.exp:
            row.append(teacher.membership_query(sequence + suffix))
        return row


class _ObservationTable(ObservationTable):
    def __init__(self):
        super().__init__()

    def is_closed(self) -> bool:
        return all(self.same_row_exists_in_red(s) for s in self.blue)

    def same_row_exists_in_red(self, blueSequence: Sequence) -> bool:
        # TODO check if this is what i need
        return any(self.observations[sequence] == self.observations[blueSequence]
                   for sequence in self.red)

    def find_inconsistency(self, alphabet: Alphabet) -> Union[TableInconsistency, None]:
        # TODO check if this is what i need
        redList = list(self.red)
        redListLength = len(redList)
        for i in range(redListLength):
            for j in range(i + 1, redListLength):
                red1 = redList[i]
                red2 = redList[j]
                if red1 != red2 and self.observations[red1] == self.observations[red2]:
                    inconsistency = self._inconsistency_between(
                        red1, red2, alphabet)
                    if inconsistency is not None:
                        return inconsistency
        return None
