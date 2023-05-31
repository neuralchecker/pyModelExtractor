from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.state import State
from pymodelextractor.teachers.teacher import Teacher
from pymodelextractor.learners.learner import Learner
from pythautomata.automata.deterministic_finite_automaton import DeterministicFiniteAutomaton as DFA
from pymodelextractor.learners.observation_table_learners.observation_table import epsilon
from pymodelextractor.learners.learning_result import LearningResult
from pymodelextractor.learners.counterexample_processing.rivest_schapire import RivestSchapire


class ObservationPackLearner(Learner):
    def __init__(self):
        # Pointer from state to node
        self.link_state_w_node = {}
        # Pointer from node to state
        self.link_node_w_state = {}
        # Transitions
        self.transitions = {}
        pass

    @property
    def _alphabet(self):
        return self._teacher.alphabet

    @property
    def _symbols(self):
        return self._teacher.alphabet.symbols

    def initialization(self) -> tuple[bool, DFA]:        
        is_final = self._teacher.membership_query(epsilon)
        starting_dfa = self.create_single_state_DFA(is_final)

        nodeRoot = ClassificationNode(epsilon)
        nodeEpsilon = ClassificationNode(epsilon, parent = nodeRoot)
        
        if is_final:
            nodeRoot.right = nodeEpsilon
        else:
            nodeRoot.left = nodeEpsilon
    
        self._tree = ClassificationTree(nodeRoot, self._teacher)
        self.link_state_w_node[starting_dfa.initial_state] = nodeEpsilon
        self.link_node_w_state[nodeEpsilon] = starting_dfa.initial_state

        # Initialize transitions pointing to root node (non-tree transitions)
        for symbol in self._symbols:
            self.transitions[(starting_dfa.initial_state, symbol)] = nodeRoot

        return (False, None)
    
    def close_transitions(self):
        for transtion in self.transitions:
            access_string, symbol = transtion
            tgt = self._tree.sift(access_string, symbol)
            self.transitions[]
        return None

    def learn(self, teacher: Teacher) -> LearningResult:
        self._teacher = teacher
        is_target_DFA, model = self.initialization()
        if not is_target_DFA:
            model = self.tentative_hypothesis()
            are_equivalent, counterexample = self._teacher.equivalence_query(model)
            while not are_equivalent:
                self.update_tree(counterexample, model)
                model = self.tentative_hypothesis()
                are_equivalent, counterexample = self._teacher.equivalence_query(model)
                if not(are_equivalent):
                    while self._teacher.membership_query(counterexample) != model.accepts(counterexample):
                        self.update_tree(counterexample, model)
                        model = self.tentative_hypothesis()               

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
       return None

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
        prev_node = None
        is_right = False
        while (not node.is_leaf()) and (not node.is_leaf()):
            d = node.string
            sd = sequence+d
            prev_node = node
            if self._ask_membership_query(sd):
                is_right = True
                node = node.right
            else:
                is_right = False
                node = node.left

        if node is None:
            new_node = ClassificationNode(sequence, parent=prev_node)
            if is_right:
                prev_node.right = new_node
            else:
                prev_node.left = new_node
            node = new_node

        self._sift_cache[sequence] = node.string

        return node.string
    
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
          