from typing import Tuple, Union

from pythautomata.base_types.sequence import Sequence
from pymodelextractor.teachers.teacher import Teacher
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.abstract.finite_automaton import FiniteAutomaton as Automaton
from pythautomata.automata.mealy_machine import MealyMachine as Mealys
from pythautomata.automata.moore_machine_automaton import MooreMachineAutomaton as MM
from pythautomata.automata.deterministic_finite_automaton import FiniteAutomataComparator as FAComparator
from pymodelextractor.teachers.pac_comparison_strategy import PACComparisonStrategy as PAC
from pythautomata.model_comparators.mealy_machine_comparison_strategy import MealyMachineComparisonStrategy as MealyComparator
from pythautomata.model_comparators.moore_machine_comparison_strategy import MooreMachineComparisonStrategy as MooreComparator

class GeneralTeacher(Teacher):
    def __init__(self, state_machine,
                comparison_strategy: Union[FAComparator, PAC, MealyComparator, MooreComparator],
                w_cache = True):
        self._state_machine = state_machine
        self._comparison_strategy = comparison_strategy
        self._cache = {}
        self._w_cache = w_cache
        super().__init__()

    @property
    def membership_queries_count(self) -> int:
        return self._membership_queries_count

    @property
    def alphabet(self) -> Alphabet:
        return self._state_machine._alphabet

    @property
    def output_alphabet(self) -> Alphabet:
        return self._state_machine.output_alphabet

    @property
    def equivalence_queries_count(self) -> int:
        return self._equivalence_queries_count

    def membership_query(self, sequence: Sequence):
        self._membership_queries_count += 1

        if not self._w_cache:
            return self._state_machine.process_query(sequence)  
        
        if sequence not in self._cache:
            result = self._state_machine.process_query(sequence)
            self._cache[sequence] = result
            return result

        return self._cache[sequence]

    def equivalence_query(self, model: Union[Automaton, MM, Mealys]) \
            -> Tuple[bool, Union[Sequence, None]]:
        self._equivalence_queries_count += 1
        counterexample = self._comparison_strategy \
            .get_counterexample_between(model, self._state_machine)
        are_equivalent = counterexample is None

        return are_equivalent, counterexample

    def reset_statistics(self) -> None:
        self._membership_queries_count = 0
        self._equivalence_queries_count = 0
