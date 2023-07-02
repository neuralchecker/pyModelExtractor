from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.state import State
from pymodelextractor.teachers.teacher import Teacher
from pymodelextractor.learners.learner import Learner
from pythautomata.automata.deterministic_finite_automaton import DeterministicFiniteAutomaton as DFA
from pymodelextractor.learners.observation_table_learners.observation_table import epsilon
from pymodelextractor.learners.learning_result import LearningResult
from pymodelextractor.learners.counterexample_processing.rivest_schapire import RivestSchapire
from pythautomata.model_exporters.dot_exporters.dfa_dot_exporting_strategy import DfaDotExportingStrategy

class ObservationPackLearner(Learner):
    def __init__(self, cex_analysis: str = 'rs'):
        # Pointer from state to node
        self.link_state_t_node = {}
        # Pointer from node to state
        self.link_node_t_state = {}
        # Outgoing transitions of a state
        self.outgoing = {}
        # Incoming transitions of a state
        self.incoming = {}
        # Open transitions set
        self.open_transitions = set()
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
    
    # CHANGE OUTGOING TO FOR #
    def learn(self, teacher: Teacher) -> LearningResult:
        self._teacher = teacher
        model = self.initialization()
        are_equivalent, counterexample = self._teacher.equivalence_query(model)
        counter = 0
        while not are_equivalent and counter < 10:
            counter += 1
            model = self.refine(model, counterexample)
            if self._tree._ask_membership_query(counterexample) == model.accepts(counterexample):
                are_equivalent, counterexample = self._teacher.equivalence_query(model)
                
        numberOfStates = len(model.states) if model is not None else 0
        info = {
            'equivalence_queries_count': self._teacher.equivalence_queries_count,
            'membership_queries_count': self._teacher.membership_queries_count,
            'discrimination_tree': self._tree
        }
        return LearningResult(model, numberOfStates, info)

    def initialization(self) -> DFA:       
        is_final = self._teacher.membership_query(epsilon)
        epsilon_state = self.create_single_state(epsilon, is_final) 

        hypothesis =  DFA(self._alphabet, epsilon_state, set([epsilon_state]), None)
        nodeRoot = ClassificationNode(epsilon)
        nodeEpsilon = ClassificationNode(epsilon, parent = nodeRoot)
        
        if is_final:
            nodeRoot.right = nodeEpsilon
        else:
            nodeRoot.left = nodeEpsilon
    
        self._tree = ClassificationTree(nodeRoot, self._teacher)
        self.link_state_t_node[hypothesis.initial_state] = nodeEpsilon
        self.link_node_t_state[nodeEpsilon] = hypothesis.initial_state

        hypothesis = self.close_transitions(hypothesis)

        return hypothesis
    
    def close_transitions(self, model: DFA) -> DFA:
        while len(self.open_transitions) > 0:
            state, symbol = self.open_transitions.pop()
            transition_aseq = state.name + Sequence([symbol])
            print("aseq", transition_aseq)
            tgt, new_state_discovered, is_final = self._tree.sift(transition_aseq)

            # New state discovered
            if new_state_discovered:
                new_state = self.create_single_state(tgt.string, is_final)
                self.link_state_t_node[new_state] = tgt
                self.link_node_t_state[tgt] = new_state

            state.transitions[symbol] = {self.link_node_t_state[tgt]}

            self.outgoing[state].add((state, symbol))
            self.incoming[self.link_node_t_state[tgt]].add((state, symbol))

        states = list(self.link_state_t_node.keys())
        for state in states:
            print(state.name, state.transitions)
        return DFA(self._alphabet, model.initial_state, 
                        set(states), None)
    
    def refine(self, model: DFA, counterexample: Sequence) -> DFA:
        u,a,v = self.analyze_inconsistency(counterexample, model)
        print("u", u, "a", a, "v", v)
        self.split(u, a ,v ,model)
        model = self.close_transitions(model)
        return model
    
    def analyze_inconsistency(self, counterexample: Sequence, hypothesis: DFA) -> \
    tuple[Sequence, Sequence, Sequence]:
        v = self.cex_analyzer.process_counterexample(counterexample, hypothesis, self._teacher)
        if counterexample[:len(counterexample) - len(v) - 1] == () or counterexample[:len(counterexample) - len(v) - 1] == []:
            u = Sequence()
        else:
            u = Sequence(counterexample[:len(counterexample) - len(v) - 1])

        a = Sequence([counterexample[len(counterexample) - len(v) - 1]])
        return (u,a,v)
    
    def split(self, u: Sequence, a: Sequence, v: Sequence, hypothesis: DFA):
        if(u == Sequence()): 
            ua = a
        else:
            ua = u + a

        q_old = self.cex_analyzer.get_end_state(hypothesis, ua)
        q_new = self.create_single_state(ua, q_old.is_final)
        self.split_leaf(q_old, q_new, v)
        self.reset_closed_transitions(q_old)

    def split_leaf(self, q_old: State, q_new: State, v: Sequence):
        print(self.link_state_t_node[q_old].string)
        old_leaf = self.link_state_t_node[q_old]
        old_leaf_parent = old_leaf.parent
        new_parent = ClassificationNode(v, parent = old_leaf_parent)
        new_leaf = ClassificationNode(q_new.name, parent = new_parent)
        
        if old_leaf_parent.left == old_leaf:
            old_leaf_parent.left = new_parent
        else:
            old_leaf_parent.right = new_parent

        if self._tree._ask_membership_query(q_old.name + v):
            new_parent.right = old_leaf
            new_parent.left = new_leaf
        else:
            new_parent.left = old_leaf
            new_parent.right = new_leaf

        self.link_state_t_node[q_new] = new_leaf
        self.link_node_t_state[new_leaf] = q_new
    
    def reset_closed_transitions(self, state: State):
        self.open_transitions = self.open_transitions.union(self.outgoing[state].union(self.incoming[state]))
        self.incoming[state] = set()
        self.outgoing[state] = set()

    def create_single_state(self, name: Sequence, is_final: bool) -> State: 
        new_state = State(name=name, is_final=is_final)
        self.outgoing[new_state] = set()
        self.incoming[new_state] = set()
        for symbol in self._symbols:
            self.open_transitions.add((new_state, symbol))
        return new_state

class ClassificationTree():
    def __init__(self, root: 'ClassificationNode', teacher: Teacher, cache_queries: bool = True):
        self._teacher = teacher
        self.root = root
        self._mq_cache = {}
        self._sift_cache = {}
        self._cache_queries = cache_queries

    def _ask_membership_query(self, sequence: Sequence) -> bool:
        if self._cache_queries:
            if sequence in self._mq_cache:
                return self._mq_cache[sequence]
        
        mq = self._teacher.membership_query(sequence)
        
        if self._cache_queries:
            self._mq_cache[sequence] = mq

        return mq

    def sift(self, sequence: Sequence) -> tuple['ClassificationNode', bool, bool]:
        node = self.root
        prev_node = None
        is_right = False
        first_mq = True
        is_final = False
        new_state_discovered = False

        while (not node.is_leaf()):
            d = node.string
            sd = sequence + d
            prev_node = node
            if self._ask_membership_query(sd):
                is_right = True
                node = node.right
            else:
                is_right = False
                node = node.left
            
            print(f"sift {d}-{sd}-{is_right}")
            if first_mq:
                first_mq = False
                is_final = is_right
                
            if node is None:
                new_state_discovered = True
                new_node = ClassificationNode(sequence, parent=prev_node)
                if is_right:
                    prev_node.right = new_node
                else:
                    prev_node.left = new_node
                node = new_node
        print("end_sift", node.string)
        return node, new_state_discovered, is_final
    
class ClassificationNode():
    def __init__(self, string: Sequence, parent: 'ClassificationNode' = None):
        self.parent = parent
        self.right = None
        self.left = None
        self.string = string

    def is_leaf(self) -> bool:
        return self.right is None and self.left is None