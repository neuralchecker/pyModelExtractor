from pythautomata.abstract.model import Model
from pythautomata.abstract.boolean_model import BooleanModel
from pythautomata.automata.wheighted_automaton_definition.weighted_automaton import WeightedAutomaton as WA
from typing import Union


class LearningResult:
    model: Union[Model, BooleanModel, WA]
    state_count: int
    info: dict

    def __init__(self, model: Union[Model, BooleanModel, WA], state_count: int, info: dict = {}):
        self.model = model
        self.state_count = state_count
        self.info = info
