import numpy as np
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.sequence import Sequence
from pythautomata.utilities.sequence_generator import SequenceGenerator


def are_within_tolerance_limit(obs1, obs2, tolerance):
    assert(len(obs1) == len(obs2))

    return np.all((abs(np.array(obs1) - np.array(obs2)) <= tolerance))


def get_test_data(alphabet: Alphabet, size: int) -> list[Sequence]:
    raise Exception('DEPRECATED')
    sq = SequenceGenerator(alphabet, 15)
    return sorted(sq.generate_words(size))


def get_partition(value, partitions):
    assert(value >= 0 and value <= 1)
    limits = np.linspace(0, 1, partitions+1)
    if value == 1: partitions-1
    for i in range(len(limits)-1):
        if limits[i]<=value and limits[i+1]>value:
            return i
      
def get_partitions(observation, partitions):
    return np.fromiter((get_partition(xi,partitions) for xi in observation), dtype=int)

def are_in_same_partition(obs1, obs2, partitions):
    assert(len(obs1) == len(obs2))
    return np.all((abs(get_partitions(obs1, partitions) - get_partitions(obs2, partitions)) == 0))