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
            end_state = self.get_end_state(hypothesis, counterexample.value[:mid])
            prefix = self.get_sequence(end_state.name)
            sec_half_counterexample = Sequence(counterexample.value[mid:])
            mq = target.membership_query(prefix + sec_half_counterexample)
            if mq == counterexample_out:
                lower = mid + 1
                if upper < lower:
                    return Sequence(sec_half_counterexample.value[1:])
            else:
                upper = mid - 1
                if upper < lower:
                    return sec_half_counterexample
                
    def get_end_state(self, hypothesis: DFA, sequenceValue: tuple[SymbolStr]) -> State:
        actual_state = hypothesis.initial_state
        if sequenceValue != ():
            for symbol in sequenceValue:
                actual_state = actual_state.next_state_for(symbol)
        return actual_state
    
    def get_sequence(self, name: str) -> Sequence:
        if name == 'ϵ':
            return Sequence()
        
        symbols = []
        for char in name:
            symbols.append(SymbolStr(char))
        
        return Sequence(symbols)
