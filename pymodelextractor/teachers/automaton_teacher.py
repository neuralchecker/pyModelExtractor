from typing import Tuple, Union
from pythautomata.abstract.finite_automaton import FiniteAutomaton
from pythautomata.base_types.sequence import Sequence
from pymodelextractor.teachers.teacher import Teacher
from pythautomata.automata.deterministic_finite_automaton import DeterministicFiniteAutomaton as DFA
from pythautomata.abstract.finite_automaton import FiniteAutomataComparator
from pythautomata.base_types.alphabet import Alphabet


class DeterministicFiniteAutomatonTeacher(Teacher):
    def __init__(self, automaton: DFA, comparison_strategy: FiniteAutomataComparator):
        self.automaton = automaton
        self._comparison_strategy = comparison_strategy
        super().__init__()

    @property
    def membership_queries_count(self) -> int:
        return self._membership_queries_count

    @property
    def alphabet(self) -> Alphabet:
        return self.automaton.alphabet

    @property
    def equivalence_queries_count(self) -> int:
        return self._equivalence_queries_count

    def membership_query(self, sequence: Sequence) -> bool:
        self._membership_queries_count += 1
        return self.automaton.accepts(sequence)

    def equivalence_query(self, model: FiniteAutomaton) -> Tuple[bool, Union[Sequence, None]]:
        self._equivalence_queries_count += 1
        counterexample = self._comparison_strategy.get_counterexample_between(model, self.automaton)
        are_equivalent = counterexample is None
        return are_equivalent, counterexample

    def reset_statistics(self) -> None:
        self._membership_queries_count = 0
        self._equivalence_queries_count = 0
