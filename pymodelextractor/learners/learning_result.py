from pythautomata.abstract.finite_automaton import FiniteAutomaton as FA


class LearningResult():
    model: FA
    state_count: int

    def __init__(self, model: FA, state_count: int):
        self.model = model
        self.state_count = state_count
