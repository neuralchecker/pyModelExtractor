from typing import Tuple, Union
import time
from pythautomata.base_types.sequence import Sequence
from pymodelextractor.teachers.teacher import Teacher
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.automata.moore_machine_automaton import MooreMachineAutomaton


class GenericTeacher(Teacher):
    def __init__(self, state_machine, comparison_strategy):
        self.state_machine = state_machine
        self._comparison_strategy = comparison_strategy
        self.log_hierachy = 'none'
        super().__init__()

    @property
    def membership_queries_count(self) -> int:
        return self._membership_queries_count

    @property
    def alphabet(self) -> Alphabet:
        return self.state_machine._alphabet

    @property
    def output_alphabet(self) -> Alphabet:
        return self.state_machine._output_alphabet

    @property
    def equivalence_queries_count(self) -> int:
        return self._equivalence_queries_count

    def membership_query(self, sequence: Sequence):
        self._membership_queries_count += 1
        
        if (type(self.state_machine)) == MooreMachineAutomaton:
            return self.state_machine.last_symbol(sequence)
        else:
            return self.state_machine.accepts(sequence)

        #return self.state_machine.process_query(sequence)

    def equivalence_query(self, model) -> Tuple[bool, Union[Sequence, None]]:
        self._equivalence_queries_count += 1
        counterexample = self._comparison_strategy.get_counterexample_between(model, self.state_machine)
        are_equivalent = counterexample is None

        return are_equivalent, counterexample

    def reset_statistics(self) -> None:
        self._membership_queries_count = 0
        self._equivalence_queries_count = 0
