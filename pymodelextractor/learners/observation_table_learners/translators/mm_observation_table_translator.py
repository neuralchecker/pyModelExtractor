from typing import Union

from pymodelextractor.learners.observation_table_learners.mm_observation_table import \
    MMObservationTable as MMOT
from pythautomata.automata.moore_machine_automaton import MooreMachineAutomaton as MM
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.moore_state import MooreState as State
from pythautomata.model_comparators.moore_machine_comparison_strategy import \
    MooreMachineComparisonStrategy as MMComparator
from symtable import Symbol

class MMObservationTableTranslator:
    epsilon = Sequence([])

    def translate(self, observation_table: MMOT, alphabet: Alphabet, outputAlphabet: Alphabet) -> MM:
        sequence_states: dict[Sequence, State] = self._get_states_for(
            observation_table.red, observation_table)
        for sequence, state in sequence_states.items():
            for symbol in alphabet.symbols:
                to_find = observation_table[sequence + symbol]
                transition_state = self._find_state_with_row(
                    to_find, observation_table, sequence_states)
                if transition_state is not None:
                    state.add_transition(symbol, transition_state)
        return MM(alphabet, outputAlphabet, sequence_states[self.epsilon], set(sequence_states.values()), MMComparator())

    def _get_states_for(self, red: set[Sequence], observation_table: MMOT) -> dict[Sequence, State]:
        sequences = [self.epsilon]

        added_rows: set[tuple[Symbol, ...]] = {tuple(observation_table[self.epsilon])}
        for sequence in red:
            if tuple(observation_table[sequence]) not in added_rows:
                sequences.append(sequence)
                added_rows.add(tuple(observation_table[sequence]))

        return {seq: State(str(seq), observation_table[seq][0]) for seq in sequences}

    def _find_state_with_row(self, row_to_find: list[Symbol], observation_table: MMOT,
                             sequence_states: dict[Sequence, State]) -> Union[State, None]:
        for sequence in sequence_states.keys():
            if observation_table[sequence] == row_to_find:
                return sequence_states[sequence]
        return None

