from pythautomata.automata.wheighted_automaton_definition.weighted_automaton import WeightedState
from pythautomata.automata.wheighted_automaton_definition.probabilistic_deterministic_finite_automaton import \
     ProbabilisticDeterministicFiniteAutomaton as PDFA
from pythautomata.model_comparators.wfa_tolerance_comparison_strategy import WFAToleranceComparator as PDFAComparator
from pythautomata.base_types.symbol import Symbol

from pymodelextractor.learners.observation_table_learners.pdfa_observation_table import PDFAObservationTable, \
     epsilon
from pymodelextractor.learners.observation_table_learners.translators.pdfa_observation_table_translator import \
     PDFAObservationTableTranslator

from collections import namedtuple


class PDFALStarColQuantObservationTableTranslator(PDFAObservationTableTranslator):
    class IntermediateState:

        Transition = namedtuple('Transition', 'prefix weight')

        def __init__(self, observation, comparator):
            self.observation = observation
            self.transitions = dict()
            self.comparator = comparator

        def belongs_to_state(self, value):
            return self.comparator.equivalent_output(value, self.observation[1])

        def has_sequence(self, sequence):
            return sequence == self.observation[0]

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
        wfa_states = self.__make_wfa_states(states)
        self.__add_wfa_transitions(states, wfa_states)
        wfa_states = set(wfa_states)
        return PDFA(observation_table.alphabet, wfa_states, terminal_symbol, PDFAComparator())

    def __make_states(self, red, comparator):
        intermediate_states = list()
        red_prefixes = list(sorted(red.keys()))
        for key in red_prefixes:
            new_intermediate_state = self.IntermediateState((key, red[key]), comparator)
            intermediate_states.append(new_intermediate_state)
        return intermediate_states

    def __add_transitions(self, observation_table, intermediate_states):
        for state in intermediate_states:
            prefix, obs = state.observation
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
            if state.has_sequence(epsilon):
                initial_value = 1
            final_value = state.observation[1][0]
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
