from pythautomata.base_types.sequence import Sequence
from pymodelextractor.teachers.teacher import Teacher
from pythautomata.automata.deterministic_finite_automaton import DeterministicFiniteAutomaton as DFA

class RivestSchapire:

    def process_counterexample(self, counterexample: Sequence, hypothesis: DFA, target: Teacher) \
    -> Sequence:
        lower = 1
        upper = len(counterexample) - 2
        counterexample_out = target.membership_query(counterexample)
        while True:
            mid = (lower + upper) / 2
            end_state = hypothesis.get_end_state(counterexample[:mid])
            prefix = end_state
            sec_half_counterexample = counterexample[mid:]
            mq = target.membership_query(prefix + sec_half_counterexample)
            if mq == counterexample_out:
                lower = mid + 1
                if upper < lower:
                    return sec_half_counterexample[1:]
            else:
                upper = mid + 1
                if upper < lower:
                    return sec_half_counterexample
