from pythautomata.automata.wheighted_automaton_definition.weighted_automaton import WeightedState
from pythautomata.automata.wheighted_automaton_definition.probabilistic_deterministic_finite_automaton import ProbabilisticDeterministicFiniteAutomaton as PDFA
from pythautomata.base_types.sequence import Sequence
from pythautomata.model_comparators.wfa_comparison_strategy import WFAComparator as PDFAComparator
from pymodelextractor.learners.observation_table_learners.pdfa_observation_table import PDFAObservationTable, \
     epsilon
from pymodelextractor.learners.observation_table_learners.translators.pdfa_observation_table_translator import \
     PDFAObservationTableTranslator

from collections import namedtuple
from pymodelextractor.utilities import pdfa_utils
import numpy as np


class PDFALStarObservationTableTranslation(PDFAObservationTableTranslator):
    class IntermediateState:

        Transition = namedtuple('Transition', 'prefix weight')

        def __init__(self, tolerance):
            self.centroid = None
            self.obs_count = 0
            self.observations = dict()
            self.transitions = dict()
            self.tolerance = tolerance

        def remove_observation(self, key):
            self.observations.pop(key)
            self.obs_count -= 1

        def reset(self):
            self.transitions = dict()
            self.centroid = sum(self.observations.values()) / len(self.observations)

        def add_observation(self, key, value):
            arr_value = np.array(value)
            self.observations[key] = arr_value
            if self.centroid is None:
                self.centroid = arr_value
            else:
                self.centroid = self.__new_centroid(arr_value)
            self.obs_count += 1

        def belongs_to_state(self, value, criterion):
            return criterion(pdfa_utils.are_within_tolerance_limit(value, obs, self.tolerance)
                             for obs in self.observations.values())

        def distance_to_centroid(self, value):
            np_value = np.array(value)
            return np.ma.sqrt(sum((np_value - self.centroid) ** 2))

        def membership_value(self, value, criterion=all):
            if self.belongs_to_state(value, criterion):
                return True, self.distance_to_centroid(value)
            else:
                return False, float('Inf')

        def has_sequence(self, sequence):
            return sequence in self.observations.keys()

        def add_transition(self, prefix, symbol, next_state_pos, weight):
            if symbol not in self.transitions:
                self.transitions[symbol] = dict()
            if next_state_pos not in self.transitions[symbol]:
                self.transitions[symbol][next_state_pos] = list()
            self.transitions[symbol][next_state_pos].append(self.Transition(prefix, weight))

        def __new_centroid(self, value):
            return (self.centroid * self.obs_count + value) / (self.obs_count + 1)

    def translate(self, observation_table: PDFAObservationTable, tolerance: float, terminal_symbol: Sequence) \
            -> PDFA:
        states = self.__make_states(observation_table.get_red_observations(), tolerance)
        self.__add_transitions(observation_table, states)
        was_deterministic = self.__make_deterministic(states, tolerance)
        while not was_deterministic:
            self.__reset_states(states)
            self.__add_transitions(observation_table, states)
            was_deterministic = self.__make_deterministic(states, tolerance)
        wfa_states = self.__make_wfa_states(states)
        self.__add_wfa_transitions(states, wfa_states)
        wfa_states = set(wfa_states)
        return PDFA(observation_table.alphabet, wfa_states, terminal_symbol, PDFAComparator())

    def __make_states(self, red, tolerance):
        intermediate_states = list()
        red_prefixes = list(sorted(red.keys()))
        for key in red_prefixes:
            self.__add_to_state(intermediate_states, (key, red[key]), tolerance)
        return intermediate_states

    def __add_to_state(self, intermediate_states, red_obs, tolerance):
        i = 0
        membership_values = np.array([])
        belongs = np.array([])
        while i < len(intermediate_states):
            intermediate_state = intermediate_states[i]
            belong, membership_value = intermediate_state.membership_value(red_obs[1])
            belongs = np.append(belongs, belong)
            membership_values = np.append(membership_values, membership_value)
            i += 1
        if np.sum(belongs) == 0:
            new_intermediate_state = self.IntermediateState(tolerance)
            new_intermediate_state.add_observation(red_obs[0], red_obs[1])
            intermediate_states.append(new_intermediate_state)
        else:
            min_dist_arg = np.argmin(membership_values)
            intermediate_states[min_dist_arg].add_observation(red_obs[0], red_obs[1])

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
                        if next_state.has_sequence(new_sequence):
                            state.add_transition(prefix, symbol.value[0], next_state_pos, obs[symbol_pos])
                            added = True
                        next_state_pos += 1
                    if not added:
                        next_state_pos = 0
                        belongs = np.array([])
                        membership_values = np.array([])
                        while next_state_pos < len(intermediate_states):
                            next_state = intermediate_states[next_state_pos]
                            belong, membership_value = \
                                next_state.membership_value(observation_table[new_sequence], criterion=any)
                            belongs = np.append(belongs, belong)
                            membership_values = np.append(membership_values, membership_value)
                            next_state_pos += 1
                        if np.sum(belongs) > 0:
                            min_dist_arg = np.argmin(membership_values)
                            state.add_transition(prefix, symbol.value[0], min_dist_arg, obs[symbol_pos])

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

    def __make_deterministic(self, states, tolerance):
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
                            intermediate_state = self.IntermediateState(tolerance)
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
