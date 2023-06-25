from pythautomata.base_types.sequence import Sequence
from pymodelextractor.teachers.teacher import Teacher
from pythautomata.automata.deterministic_finite_automaton import DeterministicFiniteAutomaton as DFA
from pythautomata.base_types.state import State
from pythautomata.base_types.symbol import SymbolStr

class RivestSchapire:

    def process_counterexample(self, counterexample: Sequence, hypothesis: DFA, target: Teacher) \
    -> Sequence:
        counterexample_out = target.membership_query(counterexample)
        lower = 1
        upper = len(counterexample) - 2
        while True:
            mid = (lower + upper) // 2
            end_state = self.get_end_state(hypothesis, counterexample[:mid])
            prefix = end_state.name
            sec_half_counterexample = counterexample.value[mid:]
            seq = self.get_sequence(prefix, sec_half_counterexample)
            print(seq)

            mq = target.membership_query(seq)
            if mq == counterexample_out:
                lower = mid + 1
                if upper < lower:
                    return sec_half_counterexample[1:]
            else:
                upper = mid - 1
                if upper < lower:
                    return sec_half_counterexample
                
    def get_end_state(self, hypothesis: DFA, sequence: Sequence) -> State:
        actual_state = hypothesis.initial_state
        if sequence != ():
            for symbol in sequence.value:
                actual_state = actual_state.next_state_for(symbol)
        return actual_state
    
    def get_sequence(self, name: str, sec_half_counterexample: tuple[SymbolStr]) -> Sequence:
        symbols = []
        for char in name:
            symbols.append(SymbolStr(char))
        
        return Sequence([y for x in [symbols, sec_half_counterexample] for y in x])
