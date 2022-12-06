from pythautomata.automata.wheighted_automaton_definition.weighted_automaton import WeightedState
from pythautomata.automata.wheighted_automaton_definition.probabilistic_deterministic_finite_automaton import \
     ProbabilisticDeterministicFiniteAutomaton as PDFA
from pythautomata.base_types.symbol import Symbol
from pythautomata.model_comparators.wfa_comparison_strategy import WFAComparator
from pythautomata.model_comparators.wfa_tolerance_comparison_strategy import WFAToleranceComparator as PDFAComparator
from pymodelextractor.learners.observation_table_learners.pdfa_observation_table import PDFAObservationTable, \
     epsilon
from pymodelextractor.learners.observation_table_learners.translators.pdfa_observation_table_translator import \
     PDFAObservationTableTranslator

from collections import namedtuple
import numpy as np


class PDFALStarQuantObservationTableTranslation(PDFAObservationTableTranslator):
    class IntermediateState:

        Transition = namedtuple('Transition', 'prefix weight')

        def __init__(self, comparator: WFAComparator):
            self.observations = dict()
            self.transitions = dict()
            self.comparator = comparator

        def remove_observation(self, key):
            self.observations.pop(key)

        def reset(self):
            self.transitions = dict()

        def add_observation(self, key, value):
            arr_value = np.array(value)
            self.observations[key] = arr_value

        def belongs_to_state(self, value):
            return self.comparator.equivalent_output(value,  list(self.observations.values())[0])

        def has_sequence(self, sequence):
            return sequence in self.observations.keys()

        def add_transition(self, prefix, symbol, next_state_pos, weight):
            if symbol not in self.transitions:
                self.transitions[symbol] = dict()
            if next_state_pos not in self.transitions[symbol]:
                self.transitions[symbol][next_state_pos] = list()
            self.transitions[symbol][next_state_pos].append(self.Transition(prefix, weight))

    def translate(self, observation_table: PDFAObservationTable, terminal_symbol: Symbol, comparator) \
            -> PDFA:
        states = self.__make_states(observation_table.get_red_observations(), comparator)
        self.__add_transitions(observation_table, states)
        # was_deterministic = self.__make_deterministic(states, comparator)
        # while not was_deterministic:
        #     self.__reset_states(states)
        #     self.__add_transitions(observation_table, states)
        #     was_deterministic = self.__make_deterministic(states, comparator)
        wfa_states = self.__make_wfa_states(states)
        self.__add_wfa_transitions(states, wfa_states)
        wfa_states = set(wfa_states)
        return PDFA(observation_table.alphabet, wfa_states, terminal_symbol, PDFAComparator())

    def __make_states(self, red, comparator):
        intermediate_states = list()
        red_prefixes = list(sorted(red.keys()))
        for key in red_prefixes:
            self.__add_to_state(intermediate_states, (key, red[key]), comparator)
        return intermediate_states

    def __add_to_state(self, intermediate_states, red_obs, comparator):
        i = 0
        added = False
        while i < len(intermediate_states) and not added:
            intermediate_state = intermediate_states[i]
            if intermediate_state.belongs_to_state(red_obs[1]):
                intermediate_state.add_observation(red_obs[0], red_obs[1])
                added = True
            i += 1
        if not added:
            new_intermediate_state = self.IntermediateState(comparator)
            new_intermediate_state.add_observation(red_obs[0], red_obs[1])
            intermediate_states.append(new_intermediate_state)

    def __add_transitions(self, observation_table, intermediate_states):
        for state in intermediate_states:
            for prefix, obs in state.observations.items():
                for symbol_pos in range(1, len(observation_table.symbols) + 1):
                    symbol = observation_table.get_suffixes()[symbol_pos]
                    new_sequence = prefix + symbol
                    next_state_pos = 0
                    added = False
                    while next_state_pos < len(intermediate_states) and not added:
                        next_state = intermediate_states[next_state_pos]
                        if next_state.has_sequence(new_sequence) or \
                                next_state.belongs_to_state(observation_table[new_sequence]):
                            state.add_transition(prefix, symbol.value[0], next_state_pos, obs[symbol_pos])
                            added = True
                        next_state_pos += 1

    def __make_wfa_states(self, intermediate_states):
        wfa_states = list()
        i = 0
        for state in intermediate_states:
            initial_value = 0
            if epsilon in state.observations.keys():
                initial_value = 1
            final_values_sum = 0
            for obs in state.observations.values():
                final_values_sum += obs[0]
            final_value = final_values_sum / len(state.observations)
            ws = WeightedState('q' + str(i), initial_value, final_value)
            wfa_states.append(ws)
            i += 1
        return wfa_states

    def __add_wfa_transitions(self, states, wfa_states):
        state_pos = 0
        for state in states:
            wfa_state = wfa_states[state_pos]
            for symbol, transitions in state.transitions.items():
                for next_state_pos, transition in transitions.items():
                    wfa_state.add_transition(symbol, wfa_states[next_state_pos], transition[0].weight)
            state_pos += 1

    def __make_deterministic(self, states, comparator):
        was_deterministic = True
        new_states = list()
        for state in states:
            obs_to_remove = set()
            for symbol, transitions in state.transitions.items():
                if len(transitions) > 1:
                    was_deterministic = False
                    keep = True
                    for next_state_pos, transition_list in transitions.items():
                        if not keep:
                            intermediate_state = self.IntermediateState(comparator)
                            for transition in transition_list:
                                if transition.prefix not in obs_to_remove:
                                    prefix = transition.prefix
                                    obs = state.observations[prefix]
                                    intermediate_state.add_observation(prefix, obs)
                                    obs_to_remove.add(prefix)
                            if intermediate_state.obs_count > 0:
                                new_states.append(intermediate_state)
                        keep = False
            for obs in obs_to_remove:
                state.remove_observation(obs)
        states.extend(new_states)
        return was_deterministic

    def __reset_states(self, states):
        for state in states:
            state.reset()
