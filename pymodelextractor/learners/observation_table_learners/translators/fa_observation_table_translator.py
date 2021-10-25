from typing import Union

from pymodelextractor.learners.observation_table_learners.observation_table import \
    ObservationTable
from pymodelextractor.learners.observation_table_learners.translators.observation_table_translator import \
    ObservationTableTranslator
from pythautomata.abstract.finite_automaton import FiniteAutomaton as FA
from pythautomata.automata.deterministic_finite_automaton import \
    DeterministicFiniteAutomaton as DFA
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.state import State


class FAObservationTableTranslator(ObservationTableTranslator):
    """FAObservationTableTranslator will translate a given observation table to a finite automaton, which will never be nondeterministic, but might not be deterministic. 
    This is because it might be lacking some transitions. Usually used for making an inbetween automaton, as 
    the one used in lambda* algorithm
    """
    epsilon = Sequence([])

    def translate(self, observation_table: ObservationTable, alphabet: Alphabet) -> FA:
        sequence_states: dict[Sequence, State] = self._get_states_for(
            observation_table.red, observation_table)
        for sequence, state in sequence_states.items():
            for symbol in alphabet.symbols:
                to_find = observation_table[sequence + symbol]
                transition_state = self._find_state_with_row(
                    to_find, observation_table, sequence_states)
                if transition_state is not None:
                    state.add_transition(symbol, transition_state)
        return DFA(alphabet, sequence_states[self.epsilon], set(sequence_states.values()), None)

    def _get_states_for(self, red: set[Sequence], observation_table: ObservationTable) -> dict[Sequence, State]:
        sequences = [self.epsilon]
        # might thorw error on set of list, easily solved by casting list to tuple

        added_rows: set[tuple[bool, ...]] = set(
            [tuple(observation_table[self.epsilon])])
        for sequence in red:
            if tuple(observation_table[sequence]) not in added_rows:
                sequences.append(sequence)
                added_rows.add(tuple(observation_table[sequence]))

        return {seq: State(str(seq), observation_table[seq][0]) for seq in sequences}

    def _find_state_with_row(self, row_to_find: list[bool], observation_table: ObservationTable,
                             sequence_states: dict[Sequence, State]) -> Union[State, None]:
        for sequence in sequence_states.keys():
            if observation_table[sequence] == row_to_find:
                return sequence_states[sequence]
        return None
