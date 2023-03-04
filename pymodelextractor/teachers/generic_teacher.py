from typing import Tuple, Union

from pythautomata.base_types.sequence import Sequence
from pymodelextractor.teachers.teacher import Teacher
from pythautomata.base_types.alphabet import Alphabet


class GenericTeacher(Teacher):
    def __init__(self, state_machine, comparison_strategy, w_cache = True):
        self.state_machine = state_machine
        self._comparison_strategy = comparison_strategy
        self.log_hierachy = 'none'
        self._cache = {}
        self._w_cache = w_cache
        super().__init__()

    @property
    def membership_queries_count(self) -> int:
        return self._membership_queries_count

    @property
    def alphabet(self) -> Alphabet:
        return self.state_machine._alphabet

    @property
    def output_alphabet(self) -> Alphabet:
        return self.state_machine.output_alphabet

    @property
    def equivalence_queries_count(self) -> int:
        return self._equivalence_queries_count

    def membership_query(self, sequence: Sequence):
        self._membership_queries_count += 1

        if not self._w_cache:
            return self.state_machine.process_query(sequence)  
        
        if sequence not in self._cache:
            result = self.state_machine.process_query(sequence)
            self._cache[sequence] = result
            return result

        return self._cache[sequence]

    def equivalence_query(self, model) -> Tuple[bool, Union[Sequence, None]]:
        self._equivalence_queries_count += 1
        counterexample = self._comparison_strategy.get_counterexample_between(model, self.state_machine)
        are_equivalent = counterexample is None

        return are_equivalent, counterexample

    def reset_statistics(self) -> None:
        self._membership_queries_count = 0
        self._equivalence_queries_count = 0
