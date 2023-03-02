from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.symbol import Symbol
from pythautomata.automata.wheighted_automaton_definition.weighted_state import WeightedState
from pythautomata.model_comparators.wfa_tolerance_comparison_strategy import WFAToleranceComparator
from pythautomata.utilities import pdfa_utils
from pythautomata.utilities.probability_partitioner import ProbabilityPartitioner
from pymodelextractor.teachers.probabilistic_teacher import ProbabilisticTeacher
from pythautomata.automata.wheighted_automaton_definition.probabilistic_deterministic_finite_automaton import \
    ProbabilisticDeterministicFiniteAutomaton as PDFA, ProbabilisticDeterministicFiniteAutomaton
from pymodelextractor.learners.observation_table_learners.observation_table import epsilon
from pymodelextractor.learners.learning_result import LearningResult
from pymodelextractor.exceptions.query_length_exceeded_exception import QueryLengthExceededException
from collections import OrderedDict
import math


class PDFAQuantizationNAryTreeLearner:
    def __init__(self, probabilityPartitioner: ProbabilityPartitioner, pre_cache_queries_for_building_hipothesis = False, check_probabilistic_hipothesis = True):
        self.probability_partitioner = probabilityPartitioner
        self._pre_cache_queries_for_building_hipothesis = pre_cache_queries_for_building_hipothesis
        self._verbose = False
        self._tree = None
        self._check_probabilistic_hipothesis = check_probabilistic_hipothesis
        pass

    @property
    def _alphabet(self):
        return self._teacher.alphabet

    @property
    def _symbols(self):
        return self._teacher.alphabet.symbols

    @property
    def _all_symbols_sorted(self):
        symbols = sorted(list(self._alphabet.symbols))
        symbols = [self.terminal_symbol] + symbols
        return symbols

    def _perform_equivalence_query(self, model):
        return self._teacher.equivalence_query(model)

    def _perform_next_token_probabilities(self, value):
        return self._teacher.next_token_probabilities(value)

    def initialization(self, verbose) -> tuple[bool, ProbabilisticDeterministicFiniteAutomaton]:
        probabilities = self._perform_next_token_probabilities(epsilon)
        starting_pdfa = self.create_single_state_PDFA(probabilities)
        are_equivalent, counterexample = self._perform_equivalence_query(starting_pdfa)
        if are_equivalent:
            self._tree = None
            return True, starting_pdfa

        next_token_probabilities_epsilon = self._perform_next_token_probabilities(epsilon)
        next_token_probabilities_counterexample = self._perform_next_token_probabilities(counterexample)
        nodeRoot = ClassificationNode(epsilon)
        nodeEpsilon = ClassificationNode(epsilon, parent=nodeRoot, probabilities=next_token_probabilities_epsilon)
        nodeCounterexample = ClassificationNode(counterexample, parent=nodeRoot,
                                                probabilities=next_token_probabilities_counterexample)

        nodeRoot.childs[tuple(next_token_probabilities_epsilon.values())] = nodeEpsilon
        nodeRoot.childs[tuple(next_token_probabilities_counterexample.values())] = nodeCounterexample

        self._tree = ClassificationTree(nodeRoot, self._teacher, self.probability_partitioner, verbose=verbose)
        return False, starting_pdfa

    def learn(self, teacher: ProbabilisticTeacher, verbose: bool = False) -> LearningResult:
        self._verbose = verbose        
        if self._pre_cache_queries_for_building_hipothesis:
            assert hasattr(teacher, 'next_token_probabilities_batch')

        self.terminal_symbol = teacher.terminal_symbol
        self._teacher = teacher
        models = []
        is_target_DFA, model = self.initialization(verbose)
        symbols = list(self._alphabet.symbols)
        if not is_target_DFA:
            for symbol in symbols:
                self._tree.sift(Sequence([symbol]))
            
            model = self.tentative_hypothesis()
            models.append(model)
            last_size = len(model.weighted_states)
            are_equivalent, counterexample = self._perform_equivalence_query(model)

            while not are_equivalent:

                if verbose: print('Size before update:', last_size)
                self.update_tree(counterexample, model)
                model = self.tentative_hypothesis()
                models.append(model)
                size = len(model.weighted_states)
                if verbose: print('Size after update:', size)
                assert size > last_size, 'Possible infinite loop'
                last_size = size
                are_equivalent, counterexample = self._perform_equivalence_query(model)

        result = self._learning_results_for(model)
        return result

    def _learning_results_for(self, model, rename_states = False):
        numberOfStates = len(model.weighted_states) if model is not None else 0
        if rename_states:
            for count, state in enumerate(model.weighted_states):
                state.name = 'q' + str(count)

        info = {
            'equivalence_queries_count': self._teacher.equivalence_queries_count,
            'last_token_weight_queries_count': self._teacher.last_token_weight_queries_count,
            'observation_tree': self._tree
        }
        return LearningResult(model, numberOfStates, info)

    def tentative_hypothesis(self) -> PDFA:        
        states = {}
        symbols = list(self._alphabet.symbols)
        symbols.sort()
        updated_tree = True
        while updated_tree:
            if self._pre_cache_queries_for_building_hipothesis:
                self._tree.cache_queries_for_building_hipothesis()

            for leaf_str, leaf in self._tree.leaves.items():
                initial_weight = 1 if leaf_str == epsilon else 0
                terminal_symbol_probability = leaf.probabilities[self.terminal_symbol]
                state = WeightedState(leaf_str, initial_weight, terminal_symbol_probability)
                states[leaf_str] = state

            for access_string, state in states.items():
                for symbol in symbols:
                    access_string_of_transition, updated_tree = self._tree.sift(access_string + symbol)
                    if updated_tree:
                        break
                    state.add_transition(symbol, states[access_string_of_transition],
                                         self._tree.leaves[access_string].probabilities[symbol])
                if updated_tree:
                    break

        comparator = WFAToleranceComparator()
        states = set(states.values())
        return PDFA(self._alphabet, states, self.terminal_symbol, comparator=comparator, check_is_probabilistic=self._check_probabilistic_hipothesis)

    def get_accessing_string(self, model: PDFA, sequence: Sequence):
        state = model.get_first_state()
        if sequence == epsilon:
            return state.name

        while len(sequence) > 0:
            state = list(state.next_states_for(sequence[0]))[0]
            sequence = sequence[1:]
        return state.name

    def update_tree(self, counterexample: Sequence, model: PDFA) -> None:
        s_i = epsilon
        gamma_j_minus_1 = epsilon
        if self._verbose: print('CE:', counterexample)
        for prefix in counterexample.get_prefixes():
            s_i_minus_1 = s_i
            s_i, updated_tree = self._tree.sift(prefix)
            s_hat_i = self.get_accessing_string(model, prefix)
            if not s_i == s_hat_i:
                internal_node_string = prefix[-1] + self._tree.lca(s_i, s_hat_i)
                self._tree.update_node(s_i_minus_1, gamma_j_minus_1, internal_node_string)
                break
            gamma_j_minus_1 = prefix

    def create_single_state_PDFA(self, probabilities: OrderedDict[Symbol, float]):
        final_weight = probabilities[self.terminal_symbol]
        probabilities.pop(self.terminal_symbol)
        initialState = WeightedState(epsilon, 1, final_weight=final_weight)
        for symbol, probability in probabilities.items():
            initialState.add_transition(symbol, initialState, probability)
        return PDFA(self._alphabet, {initialState}, self.terminal_symbol, comparator=WFAToleranceComparator(), check_is_probabilistic=self._check_probabilistic_hipothesis)


class ClassificationTree:
    unknown_leaf = "UNKNOWN"

    def __init__(self, root: 'ClassificationNode', teacher: ProbabilisticTeacher, probability_partitioner: ProbabilityPartitioner,
                 max_query_length: int = math.inf, verbose=False):
        self.leaves = dict()
        self._teacher = teacher
        self.root = root
        self.probability_partitioner = probability_partitioner
        self.inner_nodes = dict()
        self._add_leaves_and_inner_nodes()
        self._equivalence_dict = dict()
        self._next_token_probabilities_cache = dict()
        self._partitions_cache = dict()
        self._sift_cache = dict()
        self.max_query_length = max_query_length        
        self._verbose = verbose
        

    @property
    def depth(self) -> int:
        return max([x.depth for x in self.leaves.values()])

    def _add_leaves_and_inner_nodes(self):
        q = [self.root]        
        while q:
            node = q.pop()
            if node.is_leaf():
                self.leaves.update({node.string: node})
            else:
                self.inner_nodes.update({node.string: node})
                for child in node.childs.values():
                    q.append(child)

    def cache_queries_for_building_hipothesis(self):
        symbols = list(self._teacher.alphabet.symbols)
        symbols.sort()
        queries = set()
        for access_string in self.leaves.keys():
            for symbol in symbols:
                for distinguishing_string in self.inner_nodes:
                    query = access_string + symbol + distinguishing_string
                    if query not in self._next_token_probabilities_cache:
                        queries.add(query)
        if len(queries)>0:
            results = self._teacher.next_token_probabilities_batch(queries)
            self._next_token_probabilities_cache.update(results)

    def sift(self, sequence: Sequence, update = True) -> Sequence:
        if sequence in self._sift_cache:
                return self._sift_cache[sequence], False
        node = self.root
        updated_tree = False
        while not node.is_leaf():
            d = node.string
            sd = sequence + d
            sd_probabilities = self._next_token_probabilities(sd, update).values()
            child_key = self._look_for_branch(node.childs, list(sd_probabilities))
            if child_key is not None:
                node = node.childs[tuple(child_key)]
            else:
                if update:
                    node_probabilities = self._next_token_probabilities(sequence, update)
                    new_node = ClassificationNode(sequence, parent=node, probabilities=node_probabilities)
                    node.childs[tuple(sd_probabilities)] = new_node
                    self.leaves.update({new_node.string: new_node})
                    updated_tree = True
                    node = new_node
                else:
                    return ClassificationTree.unknown_leaf,False
        self._sift_cache[sequence] = node.string
        return node.string, updated_tree

    def _are_in_same_partition(self, probs1, probs2):
        partition1 = self.probability_partitioner.get_partition(probs1)
        partition2 = self.probability_partitioner.get_partition(probs2)
        return self.probability_partitioner.are_in_same_partition(partition1, partition2)

    def _look_for_branch(self, childs, probabilities):
        if tuple(probabilities) in childs:
            return probabilities
        for probs in childs.keys():
            probs = list(probs)
            if self.probability_partitioner.are_in_same_partition(probs, probabilities):
                return probs
        return None

    def _next_token_probabilities(self, sequence: Sequence, check_max_query_length = True):
        if check_max_query_length and len(sequence) > self.max_query_length:
            raise QueryLengthExceededException
        if sequence in self._next_token_probabilities_cache:
            return self._next_token_probabilities_cache[sequence]
        else:
            value = self._teacher.next_token_probabilities(sequence)
            self._next_token_probabilities_cache[sequence] = value
            return value

    def lca(self, a: Sequence, b: Sequence) -> Sequence:
        ''' lca: lowest common ancestor '''
        if not a in self.leaves:
            print('recorcholis batman')
        t1 = self.leaves[a]
        t2 = self.leaves[b]
        if t1.depth < t2.depth:
            t = t1
            t1 = t2
            t2 = t
        while t1.depth > t2.depth:
            t1 = t1.parent
        while not (t1 is t2):
            assert not ((t1 is self.root) or (t2 is self.root))
            t1 = t1.parent
            t2 = t2.parent
        return t1.string

    def get_leftmost_node(self):
        node = self.root
        while node.left is not None:
            node = node.left[0]
        return self.leaves[node.string]

    def update_node(self, node_to_be_replaced, leaf_1, distinguishing_string):
        old_node = self.leaves[node_to_be_replaced]
        old_string = old_node.string
        old_node.string = distinguishing_string
        self.inner_nodes.update({distinguishing_string: old_node})
        next_token_probabilities_node1 = self._next_token_probabilities(leaf_1)
        next_token_probabilities_node2 = self._next_token_probabilities(node_to_be_replaced)
        node_1 = ClassificationNode(leaf_1, parent=old_node, probabilities=next_token_probabilities_node1)
        node_2 = ClassificationNode(node_to_be_replaced, parent=old_node, probabilities=next_token_probabilities_node2)

        node1_cont = leaf_1 + distinguishing_string
        node1_cont_probabilities = self._next_token_probabilities(node1_cont)
        node2_cont = node_to_be_replaced + distinguishing_string
        node2_cont_probabilities = self._next_token_probabilities(node2_cont)

        old_node.childs[tuple(node1_cont_probabilities.values())] = node_1
        old_node.childs[tuple(node2_cont_probabilities.values())] = node_2
        if self._verbose:
            print("----update_node----")
            print('Old Node (new Leaf)', node_to_be_replaced)
            print('New Leaf', leaf_1)
            print(self.leaves.keys())
        self.leaves.update({
            leaf_1: node_1,
            node_to_be_replaced: node_2,
        })
        if self._verbose:
            print(self.leaves.keys())
            print("--------")
        self._update_sift_cache(old_string)

    def _update_sift_cache(self, old_string):
        keys_to_remove = []
        for seq, access_string in self._sift_cache.items():
            if access_string == old_string:
                keys_to_remove.append(seq)

        for seq in keys_to_remove:
            del self._sift_cache[seq] 

class ClassificationNode:
    def __init__(self, string: Sequence, parent: 'ClassificationNode' = None, probabilities=None):
        self.parent = parent
        self.childs = OrderedDict()
        self.string = string
        self.probabilities = probabilities
        self._depth = parent.depth + 1 if parent else 0

    @property
    def depth(self) -> int:
        return self._depth

    def is_leaf(self) -> bool:
        return len(self.childs) == 0
