import time
from pythautomata.automata.moore_machine_automaton import MooreMachineAutomaton
from pythautomata.automata_definitions.sample_moore_machines import SampleMooreMachines
from pythautomata.model_comparators.moore_machine_comparison_strategy import MooreMachineComparisonStrategy
from pythautomata.model_comparators.dfa_comparison_strategy import DFAComparisonStrategy
from pymodelextractor.learners.observation_table_learners.mm_lstar_learner import MMLStarLearner as MooreMachineLearner
from pymodelextractor.learners.observation_table_learners.lstar_learner import LStarLearner as DFALearner
from pymodelextractor.teachers.moore_machines_teacher import MooreMachineTeacher as MMTeacher
from pymodelextractor.teachers.automaton_teacher import DeterministicFiniteAutomatonTeacher as DFATeacher
from pythautomata.utilities.nicaud_mm_generator import generate_moore_machine
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.symbol import Symbol, SymbolStr
from pythautomata.utilities.nicaud_dfa_generator import generate_dfa
from pythautomata.automata.deterministic_finite_automaton import DeterministicFiniteAutomaton


def get_mm_learner():
    return MooreMachineLearner()

def get_mm_teacher(automaton: MooreMachineAutomaton) -> MMTeacher:
    return MMTeacher(automaton, MooreMachineComparisonStrategy())

def get_dfa_learner():
    return DFALearner()

def get_dfa_teacher(automaton: DeterministicFiniteAutomaton) -> DFATeacher:
    return DFATeacher(automaton, DFAComparisonStrategy())

def test_500_states_moore_machine():
    # !!!!!!!!!!
    # input_alphabet = Alphabet.from_strings(["a", "b", "c"])
    # output_alphabet = Alphabet.from_strings(["0", "1"])
    # !!!!!!!!!!

    strings = ["0", "1"]
    input_alphabet = Alphabet(frozenset(map(SymbolStr, strings)))
    strings = ["a", "b", "c"]
    output_alphabet = Alphabet(frozenset(map(SymbolStr, strings)))
    seed = 100
    automata = generate_moore_machine(input_alphabet, output_alphabet, 500, seed)
    teacher = get_mm_teacher(automata)
    start_time = time.time()
    result = get_mm_learner().learn(teacher)
    duration = time.time() - start_time
    print("MM Learner Duration for 500 states: ", duration, "s")
    assert MooreMachineComparisonStrategy().are_equivalent(
        result.model, automata)

def test_500_states_dfa():
    strings = ["0", "1"]
    input_alphabet = Alphabet(frozenset(map(SymbolStr, strings)))
    seed = 100
    automata = generate_dfa(input_alphabet, 500, seed)
    teacher = get_dfa_teacher(automata)
    start_time = time.time()
    result = get_dfa_learner().learn(teacher)
    duration = time.time() - start_time
    print("DFA Learner Duration for 500 states: ", duration, "s")
    assert DFAComparisonStrategy().are_equivalent(
        result.model, automata)

def test_1000_states_moore_machine():
    # !!!!!!!!!!
    # input_alphabet = Alphabet.from_strings(["a", "b", "c"])
    # output_alphabet = Alphabet.from_strings(["0", "1"])
    # !!!!!!!!!!

    strings = ["0", "1"]
    input_alphabet = Alphabet(frozenset(map(SymbolStr, strings)))
    strings = ["a", "b", "c"]
    output_alphabet = Alphabet(frozenset(map(SymbolStr, strings)))
    seed = 100
    automata = generate_moore_machine(input_alphabet, output_alphabet, 1000, seed)
    teacher = get_mm_teacher(automata)
    start_time = time.time()
    result = get_mm_learner().learn(teacher)
    duration = time.time() - start_time
    print("MM Learner Duration for 1000 states: ", duration, "s")
    assert MooreMachineComparisonStrategy().are_equivalent(
        result.model, automata)

def test_1000_states_dfa():
    strings = ["0", "1"]
    input_alphabet = Alphabet(frozenset(map(SymbolStr, strings)))
    seed = 100
    automata = generate_dfa(input_alphabet, 1000, seed)
    teacher = get_dfa_teacher(automata)
    start_time = time.time()
    result = get_dfa_learner().learn(teacher)
    duration = time.time() - start_time
    print("DFA Learner Duration for 1000 states: ", duration, "s")
    assert DFAComparisonStrategy().are_equivalent(
        result.model, automata)



def run():
    test_500_states_moore_machine()
    test_500_states_dfa()
    test_1000_states_moore_machine()
    test_1000_states_dfa()

if __name__ == "__main__":
    run()