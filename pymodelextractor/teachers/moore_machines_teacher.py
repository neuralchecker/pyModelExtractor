from typing import Tuple, Union
import time
from pythautomata.base_types.sequence import Sequence
from pymodelextractor.teachers.teacher import Teacher
from pythautomata.automata.moore_machine_automaton import MooreMachineAutomaton as MM
from pythautomata.model_comparators.moore_machine_comparison_strategy import MooreMachineComparisonStrategy
from pythautomata.base_types.alphabet import Alphabet


class MooreMachineTeacher(Teacher):
    def __init__(self, moore_machine: MM, comparison_strategy: MooreMachineComparisonStrategy):
        self.moore_machine = moore_machine
        self._comparison_strategy = comparison_strategy
        self.verbose = False
        super().__init__()

    @property
    def membership_queries_count(self) -> int:
        return self._membership_queries_count

    @property
    def alphabet(self) -> Alphabet:
        return self.moore_machine._input_alphabet

    @property
    def output_alphabet(self) -> Alphabet:
        return self.moore_machine._output_alphabet

    @property
    def equivalence_queries_count(self) -> int:
        return self._equivalence_queries_count

    def membership_query(self, sequence: Sequence) -> Sequence:
        self._membership_queries_count += 1
        return self.moore_machine.last_symbol(sequence)

    def equivalence_query(self, model: MM, verbose: bool = False) -> Tuple[bool, Union[Sequence, None]]:
        start_eq_time = time.time()
        self._equivalence_queries_count += 1
        counterexample = self._comparison_strategy.get_counterexample_between(model, self.moore_machine)
        are_equivalent = counterexample is None

        duration = time.time() - start_eq_time
        if (not are_equivalent):
            if self.verbose:
                print("    - Found counterexample in " + str(duration) + "s -> " + str(counterexample))
        else:
            if self.verbose:
                print("    - Made equivalence query in " + str(duration) + "s")
        
        return are_equivalent, counterexample

    def reset_statistics(self) -> None:
        self._membership_queries_count = 0
        self._equivalence_queries_count = 0
