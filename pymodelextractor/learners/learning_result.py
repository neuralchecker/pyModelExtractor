from pythautomata.abstract.finite_automaton import FiniteAutomaton as FA
from pythautomata.automata.wheighted_automaton_definition.weighted_automaton import WeightedAutomaton as WA
from typing import Union

class LearningResult():
    model: Union[FA, WA]
    state_count: int
    info: dict

    def __init__(self, model: Union[FA, WA], state_count: int, info: dict = None):
        self.model = model
        self.state_count = state_count
        self.info = info
