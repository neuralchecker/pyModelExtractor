from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.symbol import Symbol
from pythautomata.automata.wheighted_automaton_definition.weighted_state import WeightedState
from pythautomata.model_comparators.wfa_tolerance_comparison_strategy import WFAToleranceComparator as WFAComparator
from pymodelextractor.teachers.probabilistic_teacher import ProbabilisticTeacher
from pymodelextractor.learners.pdfa_learner import PDFALearner
from pythautomata.automata.wheighted_automaton_definition.probabilistic_deterministic_finite_automaton import ProbabilisticDeterministicFiniteAutomaton as PDFA
from pymodelextractor.learners.observation_table_learners.observation_table import epsilon #TODO: Fix this smelly smell https://i.pinimg.com/originals/b9/76/b7/b976b79635bf31c0d97e38297cb54db0.jpg
from pymodelextractor.learners.learning_result import LearningResult
from pythautomata.utilities import pdfa_utils
from collections import OrderedDict
import numpy as np

class PDFANAryTreeLearner(PDFALearner):
    def __init__(self):
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
        symbols = [self.terminal_symbol]+symbols
        return symbols

    def initialization(self) -> None:       
        probabilities = self._next_token_probabilities(epsilon)
        starting_pdfa = self.create_single_state_PDFA(probabilities)
        are_equivalent, counterexample = self._teacher.equivalence_query(starting_pdfa)
        if are_equivalent:
            return (True, starting_pdfa)

        next_token_probabilities_epsilon = self._next_token_probabilities(epsilon)
        next_token_probabilities_counterexample = self._next_token_probabilities(counterexample)
        nodeRoot = ClassificationNode(epsilon)
        nodeEpsilon = ClassificationNode(epsilon, parent = nodeRoot, probabilities = next_token_probabilities_epsilon)
        nodeCounterexample = ClassificationNode(counterexample, parent = nodeRoot,  probabilities = next_token_probabilities_counterexample)
        
        nodeRoot.childs[tuple(next_token_probabilities_epsilon.values())] = nodeEpsilon         
        nodeRoot.childs[tuple(next_token_probabilities_counterexample.values())] = nodeCounterexample 

        #nodeRoot.right = (nodeEpsilon, next_token_probabilities_epsilon)
        #nodeRoot.left = (nodeCounterexample, next_token_probabilities_counterexample)
        
        self._tree = ClassificationTree(nodeRoot, self._teacher)
        return (False, starting_pdfa)

    def _next_token_probabilities(self, sequence: Sequence):
        symbols = self._all_symbols_sorted
        probabilities = self._teacher.last_token_weights(sequence, symbols)
        probabilities = OrderedDict(zip(symbols, probabilities))
        return probabilities

    def learn(self, teacher: ProbabilisticTeacher, verbose: bool = False) -> LearningResult:
        self.terminal_symbol = teacher.terminal_symbol
        self._teacher = teacher
        self.tolerance = teacher.tolerance
        #self.reset()        
        #DEBUG
        models = []
        
        #END DEBUG
        is_target_DFA, model = self.initialization()
        if not is_target_DFA:
            model = self.tentative_hypothesis()
            #DEBUG
            models.append(model)
            last_size = len(model.weighted_states)

            #END DEBUG
            are_equivalent, counterexample = self._teacher.equivalence_query(model)
            while not are_equivalent:             
                
                print('Size before update:',last_size)   
                self.update_tree(counterexample, model)
                model = self.tentative_hypothesis()
                models.append(model)
                #DEBUG
                size = len(model.weighted_states)
                print('Size after update:',size)
                #if size==last_size:
                #    for m in models:
                #        out = str('./runs/')
                #        print(m._WeightedAutomaton__exporting_strategies[0])
                #       m._WeightedAutomaton__exporting_strategies[0].export(m, out)
                if size <= last_size:
                    print('ESTO SE CAE A LA MIERDA')
                assert(size>last_size)
                last_size = size
                #END DEBUG
                are_equivalent, counterexample = self._teacher.equivalence_query(model)                

        numberOfStates = len(model.weighted_states) if model is not None else 0
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
            for leaf_str, leaf in self._tree.leaves.items():
                initial_weight = 1 if leaf_str == epsilon else 0
                terminal_symbol_probability = leaf.probabilities[self.terminal_symbol]             
                state = WeightedState(leaf_str, initial_weight, terminal_symbol_probability)
                states[leaf_str] = state
            
            for access_string, state in states.items():
                for symbol in symbols:
                    access_string_of_transition, updated_tree = self._tree.sift(access_string+symbol)
                    if updated_tree:
                        break
                    state.add_transition(symbol, states[access_string_of_transition], self._tree.leaves[access_string].probabilities[symbol])
                if updated_tree:
                        break
            
        comparator = WFAComparator()
        return PDFA(self._alphabet, set(states.values()), self.terminal_symbol, comparator=comparator)
        
    def get_accessing_string(self, model: PDFA, sequence: Sequence):
        state = model.get_first_state()
        if sequence == epsilon: 
            return state.name
            
        while len(sequence)>0:
            state = list(state.next_states_for(sequence[0]))[0]
            sequence = sequence[1:]
        return state.name
        
    def update_tree(self, counterexample: Sequence, model: PDFA)-> None:        
        s_i = epsilon
        gamma_j_minus_1 = epsilon
        distinguishing_string_found = False
        print('CE:', counterexample)
        for prefix in counterexample.get_prefixes():
            s_i_minus_1 = s_i
            s_i, updated_tree = self._tree.sift(prefix)
            s_hat_i = self.get_accessing_string(model,prefix)
            if not s_i == s_hat_i:
                internal_node_string = prefix[-1] + self._tree.lca(s_i, s_hat_i)
                self._tree.update_node(s_i_minus_1, gamma_j_minus_1, internal_node_string)
                distinguishing_string_found = True
                break
            gamma_j_minus_1 = prefix
        if not distinguishing_string_found:
            #assert(False)
            #We should update left-most leaf
            #self._tree.update_leftmost_node(counterexample)
            #we should update the node s_i_minus_1
            internal_node_string = epsilon
            self._tree.update_node(s_i, counterexample, internal_node_string)

        #Some distinguishing string must have been found, if not an infinite loop occurs    
        #assert(distinguishing_string_found)

    def create_single_state_PDFA(self, probabilities: OrderedDict[Symbol, float]):
        final_weight = probabilities[self.terminal_symbol]
        probabilities.pop(self.terminal_symbol)
        initialState = WeightedState(epsilon, 1, final_weight = final_weight)
        for symbol, probability in probabilities.items():
            initialState.add_transition(symbol, initialState, probability)
        return PDFA(self._alphabet, set([initialState]), self.terminal_symbol)

class ClassificationTree():
    def __init__(self, root: 'ClassificationNode', teacher: ProbabilisticTeacher):
        self._teacher = teacher
        self.root = root
        self.add_leaves_to_dict()
        self._equivalence_dict = dict()
    
    @property
    def depth(self) -> int:
        return max([x.depth for x in self.leaves.values()])

    def add_leaves_to_dict(self):
        q = [self.root]
        self.leaves = {}
        while q:
            node = q.pop()
            if node.is_leaf():
                self.leaves.update({node.string:node})
                continue
            for child in node.childs.values():
                q.append(child)

    def sift(self, sequence: Sequence) -> Sequence:
        node = self.root
        updated_tree = False
        while not node.is_leaf():
            d = node.string
            sd = sequence+d
            sd_probabilities = self._next_token_probabilities(sd).values()
            child_key = self._look_for_branch(node.childs, list(sd_probabilities), self._teacher.tolerance)
            if child_key is not None:
                node = node.childs[tuple(child_key)]
            else:
                node_probabilities = self._next_token_probabilities(sequence)
                new_node = ClassificationNode(sequence, parent = node, probabilities = node_probabilities)
                node.childs[tuple(sd_probabilities)] = new_node
                self.leaves.update({new_node.string:new_node})
                updated_tree = True                
                node = new_node

        return node.string, updated_tree
    
    def _look_for_branch(self, childs, probabilities, tolerance):
        if tuple(probabilities) in childs:
            return probabilities
        if tuple(probabilities) in self._equivalence_dict:
            if self._equivalence_dict[tuple(probabilities)] in childs:
                return self._equivalence_dict[tuple(probabilities)] 
        t_eq_probs = []
        for probs in childs.keys():
            probs = list(probs)
            if pdfa_utils.are_within_tolerance_limit(probs,probabilities, tolerance):
                t_eq_probs.append((probs, np.ma.sqrt(sum((np.array(probs) - np.array(probabilities)) ** 2))))
        if len(t_eq_probs) == 0:
            return None
        else:
            equivalent = tuple(min(t_eq_probs, key=lambda x: x[1])[0])
            if equivalent not in self._equivalence_dict: 
                self._equivalence_dict[equivalent] = tuple(probabilities)
            self._equivalence_dict[tuple(probabilities)] = equivalent
            return equivalent


    #TODO: This should be inside the teacher abstraction or even inside the ProbabityModel abstraction
    def _next_token_probabilities(self, sequence: Sequence):
        symbols = list(self._teacher.alphabet.symbols)
        symbols.sort()
        symbols = [self._teacher.terminal_symbol]+symbols
        probabilities = self._teacher.last_token_weights(sequence, symbols)
        probabilities = OrderedDict(zip(symbols, probabilities))
        return probabilities
    # def get_distinguishing_string(self, string1, string2):
    #     queue = []
    #     queue.append(self.root)
    #     result = []
    #     while queue:
    #         element = queue.pop()
    #         if element.is_distinguishing_string(string1, string2):
    #             return element.string
    #         else:
    #             if element.right:
    #                 queue.append(element.right)
    #             if element.left:
    #                 queue.append(element.left)        
    #     return None 
    
    def lca(self, a:Sequence, b:Sequence) -> Sequence:
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

    def update_leftmost_node(self, counterexample):
        old_node = self.get_leftmost_node()
        node1_str = old_node.string
        node2_str = counterexample
        old_node.string = epsilon

        next_token_probabilities_node1 = self._next_token_probabilities(node1_str)
        next_token_probabilities_node2 = self._next_token_probabilities(node2_str)

        node_1 = ClassificationNode(node1_str, parent = old_node, probabilities = next_token_probabilities_node1)
        node_2 = ClassificationNode(node2_str, parent = old_node, probabilities = next_token_probabilities_node2)

        old_node.left = (node_1, next_token_probabilities_node1)
        old_node.right = (node_2, next_token_probabilities_node2)
        print("-----update_leftmost_node----")
        print('Old Node (new Leaf)', node1_str)
        print('New Leaf (counterexample)', node2_str)
        print(self.leaves.keys())
        self.leaves.update({
        node1_str : node_1,
        node2_str: node_2,
        })
        print(self.leaves.keys())
        print("---------")

    def update_node(self, node_to_be_replaced, leaf_1, distinguishing_string):
        old_node = self.leaves[node_to_be_replaced]
        old_node.string = distinguishing_string

        next_token_probabilities_node1 = self._next_token_probabilities(leaf_1)
        next_token_probabilities_node2 = self._next_token_probabilities(node_to_be_replaced)
        node_1 = ClassificationNode(leaf_1, parent = old_node, probabilities = next_token_probabilities_node1)
        node_2 = ClassificationNode(node_to_be_replaced, parent = old_node, probabilities = next_token_probabilities_node2)

        node1_cont = leaf_1+distinguishing_string
        node1_cont_probabilities = self._next_token_probabilities(node1_cont)
        node2_cont = node_to_be_replaced+distinguishing_string
        node2_cont_probabilities = self._next_token_probabilities(node2_cont)

        old_node.childs[tuple(node1_cont_probabilities.values())] = node_1
        old_node.childs[tuple(node2_cont_probabilities.values())] = node_2
        # if pdfa_utils.are_within_tolerance_limit(list(next_token_probabilities_node1.values()),list(node1_cont_probabilities), self._teacher.tolerance):
        #     old_node.right = (node_1, node1_cont_probabilities)
        #     old_node.left = (node_2, node2_cont_probabilities)
        # else:
        #     old_node.right = (node_2, node2_cont_probabilities)
        #     old_node.left = (node_1, node1_cont_probabilities)
        print("----update_node----")
        print('Old Node (new Leaf)', node_to_be_replaced)
        print('New Leaf', leaf_1)
        print(self.leaves.keys())
        self.leaves.update({
        leaf_1 : node_1,
        node_to_be_replaced: node_2,
        })
        print(self.leaves.keys())
        print("--------")

class ClassificationNode():
    def __init__(self, string: Sequence, parent: 'ClassificationNode' = None, probabilities  = None):
        self.parent = parent
        #self.right = None
        #self.left = None
        self.childs = OrderedDict()
        self.string = string        
        self.probabilities = probabilities
        self._depth = parent.depth+1 if parent else 0
    
    @property
    def depth(self) -> int:
        return self._depth

    def is_leaf(self) -> bool:
        return len(self.childs) == 0
    
    # def is_distinguishing_string(self, string1: Sequence, string2: Sequence) -> bool:
    #     if self.right is None or self.left is None: 
    #         return False
    #     else:
    #         return self.left.has_leaf(string1) and self.right.has_leaf(string2) or self.left.has_leaf(string2) and self.right.has_leaf(string1) 

    # def has_leaf(self, string: Sequence) -> bool:    
    #     queue = []
    #     queue.append(self)
    #     while queue:
    #         element = queue.pop()
    #         if element.is_leaf() and element.string == string:
    #             return True
    #         else:
    #             if element.right:
    #                 queue.append(element.right)[0]
    #             if element.left:
    #                 queue.append(element.left)[0]
    #     return False 
          