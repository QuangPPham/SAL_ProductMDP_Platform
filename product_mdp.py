import numpy as np

"""
Class for specifying a Product MDP.

"""

class ProductMDP():
    def __init__(self, automaton, mdp, rewards=None, labels=None, discount=None, max_iter=None, epsilon=None):
        self.automaton = automaton
        self.mdp = mdp

        if discount is None:
            self.gamma = mdp.gamma
        else:
            self.gamma = float(discount)
            assert 0.0 < self.gamma <= 1.0, "Discount must be in (0, 1]"

        if max_iter is None:
            self.max_iter = mdp.max_iter
        else:
            self.max_iter = max_iter

        if epsilon is None:
            self.epsilon = mdp.epsilon
        else:
            self.epsilon = epsilon

        # Get atomic propositions and alphabet
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
            self.P, self.R = self._computeTransition(mdp.P, self.labels)

        # Default reward is 1 for (a, (s, q)) if (s', q') is accepting, and 0 otherwise
        if rewards is not None:
            self.R = rewards

    def _computeTransition(self, mdpP, labels):
        P = tuple([np.zeros((self.SX, self.SX)) for a in range(self.A)])
        R = tuple([np.zeros(self.SX) for a in range(self.A)])

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

                            # Reward function r(sx, a): +1 for transition into accepting state, and 0 otherwise
                            if ((s,q) not in self.acc) and ((s_new, q_new) in self.acc) and (P[a][state_idx, state_new_idx] > 0):
                                R[a][state_idx] = 1.0
        return P, R

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
            V = Vprev.copy()
            policy = np.empty_like(Vprev)
            for s in range(self.SX):
                for a in range(self.A):
                    Q[a, s] = self.R[a][s] + self.gamma * self.P[a][s, :].dot(V)
                V[s] = Q[:,s].max()
                policy[s] = Q[:, s].argmax()
        else:
            for a in range(self.A):
                Q[a] = self.R[a] + self.gamma * self.P[a].dot(Vprev)

            V = Q.max(axis=0)
            policy = Q.argmax(axis=0)

        return V, Q, policy.astype(np.int32)
    
    def value_iteration(self, GS=True):
        V = np.zeros(self.SX)
        for i in range(self.max_iter):
            Vprev = V.copy() # numpy array thing
            V, Q, policy = self.Bellman_update(Vprev, GS)
            delta = np.max(abs(V - Vprev))

            if delta < self.epsilon:
                break
        
        return V, Q, policy
    
    def computePR_policy(self, policy=None):
        """
        Compute an (S,S) transition matrix and (S,) reward vector for the MDP,
        assuming actions are selected according to the policy. Policy has shape (S,)
        if deterministic, and (S,A) if stochastic.
        """
        if policy is None:
            policy = self.policy
        
        P_pi = np.empty((self.SX, self.SX))
        R_pi = np.empty(self.SX) # R(s') given s' from s,a

        if policy.ndim == 1:
            # policy is deterministic
            for a in range(self.A):
                inds = np.where(policy == a)
                ind = inds[0]
                if len(ind) > 0:
                    P_pi[ind, :] = self.P[a][ind, :]
                    R_pi[ind] = self.R[a][ind]

        elif policy.ndim == 2:
            # policy is stochastic
            P = np.array(self.P)
            for s_new in range(self.SX):
                for s in range(self.SX):
                    P_pi[s, s_new] = policy[s, :].dot(P[:, s, s_new])
                    for a in range(self.A):
                        R_pi[s_new] += policy[s, a]*self.P[a][s, s_new]*self.R[a][s]
        else:
            raise ValueError("Policy must be 1D (deterministic) or 2D (stochastic)")

        return P_pi, R_pi

    def eval(self, policy):
        """
        Evaluate value function of a given policy
        Using analytical approach
        """

        # V = PR + gPV  => (I-gP)V = PR  => V = inv(I-gP)*PR
        P_pi, R_pi = self.computePR_policy(policy)
        V_pi = np.linalg.solve((np.eye(self.SX) - self.gamma*P_pi), R_pi)

        Q_pi = np.empty((self.A, self.SX))
        for a in range(self.A):
            Q_pi[a, :] = self.R[a][:] + self.gamma * self.P[a].dot(V_pi)

        return V_pi, Q_pi

    def policy_evaluation_iterative(self, policy, max_iter = 1000, GS=True):
        """
        Iteratively evaluate the value function for a given policy.
        policy: array of shape (SX,) for deterministic, or (SX, A) for stochastic
        Returns: V_pi (value function under policy)
        """

        V = np.zeros(self.SX)
        for _ in range(max_iter):
            V_prev = V.copy()
            if policy.ndim == 1:  # deterministic
                for sx in range(self.SX):
                    a = int(policy[sx])
                    if GS:
                        V[sx] = self.R[a][sx] + self.gamma * self.P[a][sx, :].dot(V)
                    else:
                        V[sx] = self.R[a][sx] + self.gamma * self.P[a][sx, :].dot(V_prev)
            elif policy.ndim == 2:  # stochastic
                for sx in range(self.SX):
                    for a in range(self.A):
                        if GS:
                            V[sx] += policy[sx, a] * (self.R[a][sx] + self.gamma * self.P[a][sx, :].dot(V))
                        else:
                            V[sx] += policy[sx, a] * (self.R[a][sx] + self.gamma * self.P[a][sx, :].dot(V_prev))
            else:
                raise ValueError("Policy must be 1D (deterministic) or 2D (stochastic)")
            
            if np.max(np.abs(V - V_prev)) < self.epsilon:
                break

        return V
    
    def policy_improvement(self, V):
        """
        Given a value function V, improve the policy greedily.
        Returns a deterministic policy as a (SX,) array.
        """

        q_values = np.empty((self.A, self.SX))
        for a in range(self.A):
            q_values[a] = self.R[a] + self.gamma * self.P[a].dot(V)
        policy = q_values.argmax(axis=0)
   
        return policy
    
    def policy_iteration(self, iterative = True, GS = True):
        V = np.zeros(self.SX)
        policy = np.zeros_like(V)
        for i in range(self.max_iter):
            if iterative:
                V = self.policy_evaluation_iterative(policy)
            else:
                V, _ = self.eval(policy)
            old_policy = policy.copy()
            policy = self.policy_improvement(V)

            if np.array_equal(policy, old_policy):
                break
        
        return V, policy.astype(np.int32)


    def get_action(self, sx=None, s=None, q=None, policy=None):
        """
        Returns an action for state s according to the given policy.
        Supports both deterministic (1D) and stochastic (2D) policies.
        """
        if policy is None:
            policy = self.policy

        if sx is None:
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

    def rollout(self, sx=None, s=None, q=None, a=None, policy=None, depth=1):
        """
        Simulate the trajectory of the Product MDP. If action is not given, select action
        according to policy. If depth is greater than 1, select future actions based
        on the policy. Returns the array of mdp states, dfa states, product states, actions, and rewards. 

        Note that terminal state is included, but terminal action and reward are not.
        """
        if sx is None:
            sx = self.states.index((s, q))
        else:
            s, q = self.states[sx]

        if a is None:
            a = self.get_action(s, q, policy)

        s_batch, q_batch, sx_batch, a_batch, r_batch = [s], [q], [sx], [a], [self.R[a][sx]]

        for i in range(depth):
            if self.P is not None:
                probs = self.P[a_batch[i]][sx_batch[i], :] # get distribution over sx' given sx and a
                sx_new = np.random.choice(self.SX, p=probs)
                s_new, q_new = self.states[sx_new]
                a_new = self.get_action(sx=sx_new, policy=policy)
                r_new = self.R[a_new][sx_new]
            else:
                s_new, _ = self.mdp.sample_model(s_batch[i], a_batch[i])
                q_new = self.automaton.step(q_batch[i], self.labels[s_batch[i]])[0]
                sx_new = self.states.index((s_new, q_new))
                a_new = self.get_action(sx=sx_new, policy=policy)
                r_new = self.R[a_new][sx_new]

            s_batch  += [s_new]
            q_batch  += [q_new]
            sx_batch += [sx_new]
            a_batch  += [a_new]
            r_batch  += [r_new]

        return s_batch, q_batch, sx_batch, a_batch[:-1], r_batch[:-1]
    
    def sample(self, a, sx=None, s=None, q=None):
        # Sample next mdp state, dfa state, product state and reward
        if sx is None:
            sx = self.states.index((s, q))

        truncated = False
        s_list, q_list, sx_list, a_list, r_list = self.rollout(sx=sx, a=a, depth=1, policy=np.zeros(self.SX, dtype=np.int32))
        s, q, sx, r = s_list[-1], q_list[-1], sx_list[-1], r_list[-1]
        if (s, q) in self.final:
            truncated = True
        return s, q, sx, r, truncated