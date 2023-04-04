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
from pythautomata.base_types.symbol import Symbol

class PartialDFATranslator(ObservationTableTranslator):
    # This partial observation table translator is non deterministic due to red being an 
    #   unordered set.
    # Iterating through an unordered set in python is non deterministic and leads to 
    #   different states added on final automaton resulting in a different automaton.

    hole_state = State("Hole", False)
    epsilon = Sequence([])

    def translate(self, observation_table: Union[ObservationTable, GeneralObservationTable],
                  alphabet: Alphabet, output_alphabet: Alphabet = None) \
                    -> Union[BooleanModel, Model]:
        
        states = self.create_states(observation_table)

        for seq, state in states:
            for suffix in alphabet.symbols:
                if (seq+suffix) in (observation_table.observations):
                    next_state_value = observation_table[seq + suffix]
                    next_state = self.find_state(states, observation_table, next_state_value)

                    if next_state is None:
                        state.add_hole_transition(self.hole_state)
                    else:
                        state.add_transition(suffix, next_state)
                else:
                    state.add_hole_transition(self.hole_state)
        initial_state = states[0][1] if states else self.hole_state
        return DFA(alphabet, initial_state, {state for _, state in states}, 
                   DFAComparator, hole=self.hole_state)

    def create_states(self, observation_table) -> list[tuple[Sequence, State]]:
        red_values = set()
        states = []
        if self.epsilon in observation_table.observations:
            states.append((self.epsilon, State(str(self.epsilon), observation_table[self.epsilon][0])))
        for red_seq in observation_table.red:
            if red_seq in observation_table.observations:
                red_value = tuple(observation_table[red_seq])
                if red_value not in red_values:
                    state = State(str(red_seq), red_value[0])
                    states.append((red_seq, state))
                    red_values.add(red_value)
        
        return states
    
    def find_state(self, states: list[tuple[Sequence, State]], 
                   observation_table: Union[ObservationTable, GeneralObservationTable],
                   value: Union[list[bool], list[Symbol]]) -> Union[State, None]:
        
        for sequence, state in states:
            if value == observation_table[sequence]:
                return state
        
        return None
