import numpy as np
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.sequence import Sequence
from pythautomata.utilities.sequence_generator import SequenceGenerator


def are_within_tolerance_limit(obs1, obs2, tolerance):
    if len(obs1) != len(obs2):
        print("Wrong length")
        return False
    return np.all((abs(np.array(obs1) - np.array(obs2)) <= tolerance))


def get_test_data(alphabet: Alphabet, size: int) -> list[Sequence]:
    raise Exception('DEPRECATED')
    sq = SequenceGenerator(alphabet, 15)
    return sorted(sq.generate_words(size))
