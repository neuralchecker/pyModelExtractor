from pythautomata.abstract.finite_automaton import FiniteAutomaton as FA
from pythautomata.automata.wheighted_automaton_definition.weighted_automaton import WeightedAutomaton as WA
from pythautomata.automata.moore_machine_automaton import MooreMachineAutomaton as MM
from typing import Union


class LearningResult:
    model: Union[FA, WA, MM]
    state_count: int
    info: dict

    def __init__(self, model: Union[FA, WA, MM], state_count: int, info: dict = {}):
        self.model = model
        self.state_count = state_count
        self.info = info
