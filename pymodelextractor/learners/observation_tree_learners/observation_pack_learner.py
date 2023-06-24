from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.state import State
from pymodelextractor.teachers.teacher import Teacher
from pymodelextractor.learners.learner import Learner
from pythautomata.automata.deterministic_finite_automaton import DeterministicFiniteAutomaton as DFA
from pymodelextractor.learners.observation_table_learners.observation_table import epsilon
from pymodelextractor.learners.learning_result import LearningResult
from pymodelextractor.learners.counterexample_processing.rivest_schapire import RivestSchapire


class ObservationPackLearner(Learner):
    def __init__(self, cex_analysis: str = 'rs'):
        # Pointer from state to node
        self.link_state_t_node = {}
        # Transitions
        self.transitions = {}
        # Outgoing transitions of a state
        self.outgoing = {}
        # Open transitions set
        self.open_transitions = {}
        if cex_analysis == 'rs':
            self.cex_analyzer = RivestSchapire()
        else:
            raise NotImplementedError('Counterexample analysis method not implemented')
        pass

    @property
    def _alphabet(self):
        return self._teacher.alphabet

    @property
    def _symbols(self):
        return self._teacher.alphabet.symbols

    def initialization(self) -> 'ClassificationNode':       
        is_final = self._teacher.membership_query(epsilon)
        hypothesis = self.create_single_state(is_final) 

        nodeRoot = ClassificationNode(epsilon)
        nodeEpsilon = ClassificationNode(epsilon, parent = nodeRoot)
        
        if is_final:
            nodeRoot.right = nodeEpsilon
        else:
            nodeRoot.left = nodeEpsilon
    
        self._tree = ClassificationTree(nodeRoot, self._teacher)
        self.link_state_t_node[hypothesis.initial_state] = nodeEpsilon

        # Initialize transitions pointing to root node (non-tree transitions)
        # and so add them to open transitions set
        for symbol in self._symbols:
            self.transitions[(hypothesis.initial_state, symbol)] = (nodeRoot, None)
            self.open_transitions.add(hypothesis.initial_state.name, symbol)

        self.close_transitions()

        return self._tree
    
    def close_transitions(self):
        while len(self.open_transition) > 0:
            transition = self.open_transition.pop()
            transition_aseq = transition[0]+transition[1]
            tgt, new_state_discovered, is_final = self._tree.sift(transition_aseq)
            self.transitions[transition] = tgt.string
            # New state discovered
            if new_state_discovered:
                for symbol in self._symbols:
                    self.open_transitions.add(tgt, symbol)
                new_state = self.create_single_state(tgt.string, is_final)
                self.link_state_t_node[new_state] = tgt
        

    def learn(self, teacher: Teacher) -> LearningResult:
        self._teacher = teacher
        model, self._tree = self.initialization()
        are_equivalent, counterexample = self._teacher.equivalence_query(model)
        while not are_equivalent:
            self.refine()
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
    
    def create_transitions_for_new_state(self, q_new: Sequence):
        for symbol in self._symbols:
            self.transitions[q_new, symbol] = (self._tree.root, None)
            self.open_transitions.add(q_new, symbol)
    
    def refine(self):
        self.analyze_inconsistency()
        self.split()
        self.close_transitions()
        return None
    
    def analyze_inconsistency(self, cex: Sequence) -> tuple(Sequence, Sequence, Sequence):
        v = self.cex_analyzer.analyze(cex)
        u = cex[:-len(v)]
        a = cex[-len(v):]
        return u,a,v
    
    def split(self, u: Sequence, a: Sequence, v: Sequence):
        q_old = self.get_state_sequence(u+a)
        q_new = self.create_single_state(u+a, False)
        self.create_transitions_for_new_state(q_new)
        self.split_leaf(q_old, q_new, v)
    
    def split_leaf(self, q_old: State, q_new: State, v: Sequence):
        old_leaf = self.link_state_t_node[q_old]
        old_leaf_parent = old_leaf.parent
        new_parent = ClassificationNode(v, parent = old_leaf_parent)
        new_leaf = ClassificationNode(q_new.name, parent = new_parent)
    
        if self._tree._ask_membership_query(q_old.name+v):
            new_parent.right = old_leaf
            new_parent.left = new_leaf
        else:
            new_parent.left = old_leaf
            new_parent.right = new_leaf

        self.link_state_t_node[q_new] = new_leaf

    def get_state(self, sequence: Sequence) -> State:
        # Getting first state
        state = list(self.link_state_t_node.keys())[0]
        if sequence[0] == epsilon:
            sequence = sequence[1:]
                        
        while len(sequence)>0:
            node_state = self.transitions[(state, sequence[0])]
            state = node_state[1]
            sequence = sequence[1:]
        return state
    
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
             
    def create_single_state(self, name: str, is_final: bool) -> State: 
        return State(name, is_final=is_final)

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

    def sift(self, sequence: Sequence) -> tuple('ClassificationNode', bool, bool):
        node = self.root
        prev_node = None
        is_right = False
        first_mq = True
        is_final = False
        new_state_discovered = False

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

            if first_mq and is_right:
                is_final = True
            elif first_mq and not is_right:
                is_final = False

        if node is None:
            new_state_discovered = True
            new_node = ClassificationNode(sequence, parent=prev_node)
            if is_right:
                prev_node.right = new_node
            else:
                prev_node.left = new_node
            node = new_node

        return node, new_state_discovered, is_final
    
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
          