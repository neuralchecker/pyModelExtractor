from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.state import State
from pymodelextractor.teachers.teacher import Teacher
from pymodelextractor.learners.learner import Learner
from pythautomata.automata.deterministic_finite_automaton import DeterministicFiniteAutomaton as DFA
from pymodelextractor.learners.observation_table_learners.observation_table import epsilon
from pymodelextractor.learners.learning_result import LearningResult


class KearnsVaziraniLearner(Learner):
    def __init__(self):
        pass

    @property
    def _alphabet(self):
        return self._teacher.alphabet

    @property
    def _symbols(self):
        return self._teacher.alphabet.symbols

    def initialization(self, cache_in_tree: bool) -> tuple[bool, DFA]:        
        is_final = self._teacher.membership_query(epsilon)
        starting_dfa = self.create_single_state_DFA(is_final)
        are_equivalent, counterexample = self._teacher.equivalence_query(starting_dfa)
        if are_equivalent:
            return (True, starting_dfa)

        nodeRoot = ClassificationNode(epsilon)
        nodeEpsilon = ClassificationNode(epsilon, parent = nodeRoot)
        nodeCounterexample = ClassificationNode(counterexample, parent = nodeRoot)
        
        if is_final:
            nodeRoot.right = nodeEpsilon
            nodeRoot.left = nodeCounterexample
        else:
            nodeRoot.right = nodeCounterexample
            nodeRoot.left = nodeEpsilon
        self._tree = ClassificationTree(nodeRoot, self._teacher, cache_in_tree)
        return (False, None)

    def learn(self, teacher: Teacher, cache_in_tree: bool = True) -> LearningResult:
        self._teacher = teacher
        is_target_DFA, model = self.initialization(cache_in_tree)
        if not is_target_DFA:
            model = self.tentative_hypothesis()
            are_equivalent, counterexample = self._teacher.equivalence_query(model)
            while not are_equivalent:
                self.update_tree(counterexample, model)
                model = self.tentative_hypothesis()
                are_equivalent, counterexample = self._teacher.equivalence_query(model)                

        numberOfStates = len(model.states) if model is not None else 0
        info = {
            'equivalence_queries_count': self._teacher.equivalence_queries_count,
            'membership_queries_count': self._teacher.membership_queries_count,
            'observation_tree': self._tree
        }
        return LearningResult(model, numberOfStates, info)

    def tentative_hypothesis(self) -> DFA:
        states = {}
        for leaf in self._tree.leaves:
            is_final = self._tree._ask_membership_query(leaf)
            state = State(leaf, is_final)
            states[leaf] = state
        
        for access_string, state in states.items():
            for symbol in self._symbols:
                access_string_of_transition = self._tree.sift(access_string+symbol)
                state.add_transition(symbol, states[access_string_of_transition])
        
        return DFA(self._alphabet, states[epsilon], set(states.values()), None)
        
    def get_accessing_string(self, model: DFA, sequence: Sequence):
        state = min(model.initial_states)
        if sequence == epsilon: 
            return state.name
            
        while len(sequence)>0:
            state = state.next_state_for(sequence[0])
            sequence = sequence[1:]
        return state.name
        
    def update_tree(self, counterexample: Sequence, model: DFA)-> None:        
        s_i = epsilon
        gamma_j_minus_1 = epsilon
        distinguishing_string_found = False
        for prefix in counterexample.get_prefixes():
            s_i_minus_1 = s_i
            s_i = self._tree.sift(prefix)
            s_hat_i = self.get_accessing_string(model,prefix)
            if not s_i == s_hat_i:
                internal_node_string = prefix[-1] + self._tree.lca(s_i, s_hat_i)
                self._tree.update_node(s_i_minus_1, gamma_j_minus_1, internal_node_string)
                distinguishing_string_found = True
                break
            gamma_j_minus_1 = prefix
        #Some distinguishing string must have been found, if not an infinite loop occurs    
        assert(distinguishing_string_found, 'Some distinguishing string must have been found, if not an infinite loop occurs')

    def create_single_state_DFA(self, is_final: bool):
        epsilonState = State(epsilon, is_final=is_final)
        for symbol in self._symbols:
            epsilonState.add_transition(symbol, epsilonState)
        return DFA(self._alphabet, epsilonState, set([epsilonState]), None)

class ClassificationTree():
    def __init__(self, root: 'ClassificationNode', teacher: Teacher, cache_queries: bool = True):
        self._teacher = teacher
        self.root = root
        self.add_leaves_to_dict()
        self._mq_cache = {}
        self._sift_cache = {}
        self._cache_queries = cache_queries
  
    def add_leaves_to_dict(self):
        q = [self.root]
        self.leaves = {}
        while q:
            node = q.pop()
            if node.is_leaf():
                self.leaves.update({node.string:node})
                continue
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)

    def _ask_membership_query(self, sequence: Sequence) -> bool:
        if self._cache_queries:
            if sequence in self._mq_cache:
                return self._mq_cache[sequence]
        
        mq = self._teacher.membership_query(sequence)
        
        if self._cache_queries:
            self._mq_cache[sequence] = mq

        return mq

    def sift(self, sequence: Sequence) -> Sequence:
        if self._cache_queries:
            if sequence in self._sift_cache:
                return self._sift_cache[sequence]

        node = self.root
        while not node.is_leaf():
            d = node.string
            sd = sequence+d
            if self._ask_membership_query(sd):
                node = node.right
            else:
                node = node.left

        if self._cache_queries:
            self._sift_cache[sequence] = node.string

        return node.string

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

    def update_node(self, node_to_be_replaced, leaf_1, distinguishing_string):
        old_node = self.leaves[node_to_be_replaced]
        old_string = old_node.string
        old_node.string = distinguishing_string

        node_1 = ClassificationNode(leaf_1, parent = old_node)
        node_2 = ClassificationNode(node_to_be_replaced, parent = old_node)

        if self._ask_membership_query(leaf_1+distinguishing_string):
            old_node.right = node_1
            old_node.left = node_2
        else:
            old_node.right = node_2
            old_node.left = node_1  

        self.leaves.update({
        leaf_1 : node_1,
        node_to_be_replaced: node_2,
        })
        if self._cache_queries:
            self._update_sift_cache(old_string)

    def _update_sift_cache(self, old_string):
        keys_to_remove = []
        for seq, access_string in self._sift_cache.items():
            if access_string == old_string:
                keys_to_remove.append(seq)

        for seq in keys_to_remove:
            del self._sift_cache[seq]    

class ClassificationNode():
    def __init__(self, string: Sequence, parent: 'ClassificationNode' = None):
        self.parent = parent
        self.right = None
        self.left = None
        self.string = string
        self._depth = parent.depth+1 if parent else 0
    
    @property
    def depth(self) -> int:
        return self._depth

    def is_leaf(self) -> bool:
        return self.right is None and self.left is None
    
    # def is_distinguishing_string(self, string1: Sequence, string2: Sequence) -> bool:
    #     if self.right is None or self.left is None: 
    #         return False
    #     else:
    #         return self.left.has_leaf(string1) and self.right.has_leaf(string2) or self.left.has_leaf(string2) and self.right.has_leaf(string1) 

    def has_leaf(self, string: Sequence) -> bool:    
        queue = []
        queue.append(self)
        while queue:
            element = queue.pop()
            if element.is_leaf() and element.string == string:
                return True
            else:
                if element.right:
                    queue.append(element.right)
                if element.left:
                    queue.append(element.left)        
        return False 
          