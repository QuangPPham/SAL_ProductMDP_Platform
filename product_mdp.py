import numpy as np

"""
Class for specifying a Product MDP.

"""

class ProductMDP():
    def __init__(self, automaton, mdp, rewards=None, labels=None):
        self.automaton = automaton
        self.mdp = mdp
        self.gamma = mdp.gamma
        self.max_iter = mdp.max_iter
        self.epsilon = mdp.epsilon

        self.aps = automaton.aps
        self.alphabet = automaton.labels

        # Shapes
        self.S = mdp.S
        self.A = mdp.A
        self.Q = automaton.shape[1]
        self.SX = self.S * self.Q

        # states[i] -> tuple (s[floor(i/Q)], q[i % Q])
        self.states = [(s, q) for s in range(self.S) for q in range(self.Q)]
                    
        # Labeling function
        if labels is not None:
            self.labels = labels

        # Initial state of the product MDP
        self.state0 = (mdp.s0, automaton.states.index(automaton.q0))

        # final states
        self.acc = []
        self.sink = []

        for q in range(self.Q):
            if automaton.states[q] in automaton.acc:
                self.acc += [(s,q) for s in range(self.S)]
            elif automaton.states[q] in automaton.sink:
                self.sink += [(s,q) for s in range(self.S)]

        self.final = self.acc + self.sink
        
        # Dynamics
        if mdp.P is not None:
            self.P = self._computeTransition(mdp.P, self.labels)

        # Default reward is 1 if q is accepting, and 0 otherwise
        if rewards is not None:
            self.R = rewards
        else:
            self.R = tuple([np.zeros(self.SX) for a in range(self.A)])
            for s in range(self.S):
                for q in range(self.Q):
                    if (s,q) in self.acc:
                        state_idx = self.states.index((s, q))
                        for a in range(self.A):
                            self.R[a][state_idx] = 1.0

    def _computeTransition(self, mdpP, labels):
        P = tuple([np.zeros((self.SX, self.SX)) for a in range(self.A)])

        for a in range(self.A):
            for s in range(self.S):
                for s_new in range(self.S):
                    for q in range(self.Q):
                        for q_new in range(self.Q):
                            state_idx = self.states.index((s, q))
                            state_new_idx = self.states.index((s_new, q_new))
                            # P(sx, a, sx') = P(s, a, s') if (s, q) not in final states and q' = delta(L(s))
                            q_new_actual = self.automaton.step(q, labels[s])[0]
                            if ((s,q) not in self.final) and (q_new == q_new_actual):
                                P[a][state_idx, state_new_idx] = mdpP[a][s, s_new]
                            # sx in final states -> do not go to new states
                            elif ((s,q) in self.final):
                                P[a][state_idx, state_idx] = 1
        return P

    def Bellman_update(self, Vprev, GS=True):
        """
        One step of the value iteration update using either Jacobi or Gauss-Seidel method.
        Returns value and action-value functions, and improved epsilon-greedy policy
        """

        try:
            assert Vprev.shape in ((self.SX,), (1, self.SX)), "V is not the right shape (Bellman operator)."
        except AttributeError:
            raise TypeError("V must be a numpy array or matrix.")

        Q = np.empty((self.A, self.SX))

        if GS:
            for s in range(self.SX):
                for a in range(self.A):
                    Q[a, s] = self.R[a][s] + self.gamma * self.P[a][s, :].dot(Vprev)
        else:
            for a in range(self.A):
                Q[a] = self.R[a] + self.gamma * self.P[a].dot(Vprev)
            
        V = Q.max(axis=0)
        policy = Q.argmax(axis=0)

        return V, Q, policy
    
    def value_iteration(self):
        V = np.zeros(self.SX)
        for i in range(self.max_iter):
            Vprev = V.copy() # numpy array thing
            V, Q, policy = self.Bellman_update(Vprev)
            delta = np.max(abs(V - Vprev))

            if delta < self.epsilon:
                break
        
        return V, Q, policy
    
    def get_action(self, s, q, policy=None):
        """
        Returns an action for state s according to the given policy.
        Supports both deterministic (1D) and stochastic (2D) policies.
        """
        if policy is None:
            policy = self.policy

        sx = self.states.index((s, q))

        if policy.ndim == 1:
            # policy is deterministic
            a_new = policy[sx]
        elif policy.ndim == 2:
            # policy is stochastic
            a_new = np.random.choice(self.A, p=policy[sx])
        else:
            raise ValueError("Policy must be 1D (deterministic) or 2D (stochastic)")
        
        return a_new

    def rollout(self, s, q, a=None, policy=None, depth=1):
        """
        Simulate the trajectory of the Product MDP. If action is not given, select action
        according to policy. If depth is greater than 1, select future actions based
        on the policy. Returns the array of mdp states, dfa states, product states, actions, and rewards. 

        Note that terminal state is included, but terminal action and reward are not.
        """
        sx = self.states.index((s, q))

        if a is None:
            a = self.get_action(s, q, policy)

        s_batch, q_batch, sx_batch, a_batch, r_batch = [s], [q], [sx], [a], [self.R[a][sx]]

        for i in range(depth):
            if self.P is not None:
                probs = self.P[a_batch[i], sx_batch[i], :] # get distribution over sx' given sx and a
                sx_new = np.random.choice(self.SX, p=probs)
                s_new, q_new = self.states[sx_new]
                a_new = self.get_action(s_new, q_new, policy)
                r_new = self.R[a_new][sx_new]
            else:
                s_new, _ = self.mdp.sample_model(s_batch[i], a_batch[i])
                q_new = self.automaton.step(q_batch[i], self.labels[s_batch[i]])[0]
                sx_new = self.states.index((s_new, q_new))
                a_new = self.get_action(s_new, q_new, policy)
                r_new = self.R[a_new][sx_new]

            s_batch  += [s_new]
            q_batch  += [q_new]
            sx_batch += [sx_new]
            a_batch  += [a_new]
            r_batch  += [r_new]

        return s_batch, q_batch, sx_batch, a_batch[:-1], r_batch[:-1]
    
    def sample(self, s, q, a):
        # Sample next mdp state, dfa state, product state and reward
        s_list, q_list, sx_list, a_list, r_list = self.rollout(s, q, a, depth=1)
        return s_list[-1], q_list[-1], sx_list[-1], r_list[-1]