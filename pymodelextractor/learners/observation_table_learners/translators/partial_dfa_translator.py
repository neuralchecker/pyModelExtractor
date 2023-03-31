from typing import Union
from pymodelextractor.learners.observation_table_learners.translators.\
    observation_table_translator import ObservationTableTranslator
from pymodelextractor.learners.observation_table_learners.observation_table \
    import ObservationTable
from pymodelextractor.learners.observation_table_learners.general_observation_table \
    import GeneralObservationTable
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.abstract.boolean_model import BooleanModel
from pythautomata.abstract.model import Model
from pythautomata.base_types.state import State
from pythautomata.base_types.sequence import Sequence
from pythautomata.automata.deterministic_finite_automaton import \
    DeterministicFiniteAutomaton as DFA
from pythautomata.model_comparators.dfa_comparison_strategy import DFAComparisonStrategy \
    as DFAComparator

class PartialDFATranslator(ObservationTableTranslator):
    hole_state = State("Hole", False)
    epsilon = Sequence([])

    def translate(self, observation_table: Union[ObservationTable, GeneralObservationTable],
                  alphabet: Alphabet, output_alphabet: Alphabet = None) \
                    -> Union[BooleanModel, Model]:
        
        states = self.create_states(observation_table)

        for seq, state in states:
            for suffix in alphabet.symbols:
                next_state_value = observation_table[seq + suffix]
                next_state = self.find_state(states, observation_table, next_state_value)

                if next_state is None:
                    state.add_hole_transition(self.hole_state)
                else:
                    state.add_transition(suffix, next_state)
        
        return DFA(alphabet, states[0][1], {state for _, state in states}, 
                   DFAComparator, hole=self.hole_state)

    def create_states(self, observation_table) -> list[tuple[Sequence, State]]:
        red_values = set()
        states = [(self.epsilon, State(str(self.epsilon), observation_table[self.epsilon][0]))]
        for red_seq in observation_table.red:
            if red_seq not in red_values:
                values = observation_table[red_seq]
                state = State(str(red_seq), values[0])
                states.append((red_seq, state))
                red_values.add(tuple(values))
        
        return states
    
    def find_state(self, states: list[tuple[Sequence, State]], 
                   observation_table: Union[ObservationTable, GeneralObservationTable],
                   value: list[bool]) -> Union[State, None]:
        
        for sequence, state in states:
            if value == observation_table[sequence]:
                return state
        
        return None
