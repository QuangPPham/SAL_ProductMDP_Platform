import numpy as np
import cvxpy as cp
from mdp import MDP

class ProductMDP(MDP):
    def __init__(self, automaton, labels=None, transitions=None, sample_model=None, rewards=None, horizon=None, discount=1.0, epsilon=1e-6, max_iter=10000):
        """
        Class for specifying a Product MDP.
        See MDP class for details on transitions and rewards
        See DFA class for details on the automaton
        The labels is a dictionary containing the state of each atomic proposition
            e.g: {'a': True, 'b': False}.
        """
        
        super().__init__(transitions, sample_model, rewards, horizon, discount, epsilon, max_iter)
        self.automaton = automaton

        # Get atomic propositions and alphabet
        self.aps = automaton.aps
        self.alphabet = automaton.labels

        # Shapes
        self.mdpS = self.S
        self.Q = automaton.shape[1]
        self.S = self.mdpS * self.Q

        # states[i] -> tuple (s[floor(i/Q)], q[i % Q])
        self.states = [(s, q) for s in range(self.mdpS) for q in range(self.Q)]
                    
        # Labeling function
        if labels is not None:
            self.labels = labels

        # final states
        self.acc = []
        self.sink = []

        for q in range(self.Q):
            if automaton.states[q] in automaton.acc:
                self.acc += [(s,q) for s in range(self.mdpS)]
            elif automaton.states[q] in automaton.sink:
                self.sink += [(s,q) for s in range(self.mdpS)]

        self.final = self.acc + self.sink
        
        # Dynamics
        if self.P is not None:
            self.P, self.R = self._computeTransition(self.P, self.labels)

        # Default reward is 1 for (a, (s, q)) if (s', q') is accepting, and 0 otherwise
        if rewards is not None:
            self.R = rewards

    def _computeTransition(self, mdpP, labels):
        P = tuple([np.zeros((self.S, self.S)) for a in range(self.A)])
        R = tuple([np.zeros(self.S) for a in range(self.A)])

        for a in range(self.A):
            for s in range(self.mdpS):
                for s_new in range(self.mdpS):
                    for q in range(self.Q):
                        state_idx = self.states.index((s, q))

                        # P(sx, a, sx') = P(s, a, s') if (s, q) not in final states and q' = delta(L(s))
                        if ((s,q) not in self.final):
                            q_new = self.automaton.step(q, labels[s])[0]
                            state_new_idx = self.states.index((s_new, q_new))
                            P[a][state_idx, state_new_idx] = mdpP[a][s, s_new]

                            # Reward function r(sx, a): +1 for transition into accepting state, and 0 otherwise
                            if ((s,q) not in self.acc) and ((s_new, q_new) in self.acc) and (P[a][state_idx, state_new_idx] > 0):
                                R[a][state_idx] = 1.0

                        # q is accepting/sink -> do not go to new logic state
                        elif ((s,q) in self.final):
                            state_new_idx = self.states.index((s_new, q))
                            P[a][state_idx, state_new_idx] = mdpP[a][s, s_new]

        return P, R


    def Bellman_update_reachability(self, Vprev):
        """
        One step of the value iteration update using either Jacobi or Gauss-Seidel method.
        Returns value and action-value functions, and improved epsilon-greedy policy
        """

        try:
            assert Vprev.shape in ((self.S,), (1, self.S)), "V is not the right shape (Bellman operator)."
        except AttributeError:
            raise TypeError("V must be a numpy array or matrix.")

        Q = np.empty((self.A, self.S))
        V = Vprev.copy()

        for s in range(self.S):
            (mdp_s, q) = self.states[s]
            if (mdp_s, q) in self.acc:
                V[s] = 1.0
            elif (mdp_s, q) in self.sink:
                V[s] = 0.0
            else:
                for a in range(self.A):
                    Q[a, s] = self.P[a][s, :].dot(V)
                V[s] = Q[:,s].max()

        return V, Q
    

    def value_iteration_reachability(self):
        """
        Perform value iteration. Returns value function, action-value function,
        and greedy policy. For finite-horizon, returns a list of V, Q, and policies
        for each timestep, where V[t], Q[t], and policy[t] is for timestep t.
        """

        # --- For finite-horizon --- #
        if self.T is not None:
            V = [np.zeros(self.S) for t in range(self.T+1)]
            Q = [np.empty((self.A, self.S)) for t in range(self.T)]
            policy = [np.empty(self.S) for t in range(self.T)]

            for t in reversed(range(self.T)):
                    V[t], Q[t] = self.Bellman_update_reachability(V[t+1].copy()) # update V[t] from V[t+1]

        # --- For infinite-horizon --- #
        else:
            V = np.zeros(self.S)
            for i in range(self.max_iter):
                Vprev = V.copy() # numpy array thing
                V, Q = self.Bellman_update_reachability(Vprev)
                delta = np.max(abs(V - Vprev))

                if delta < self.epsilon:
                    policy = self.extract_policy(V)
                    break
            
        return V, Q, policy


    def linear_program_reachability(self):
        """
        Linear program to solve for the maximum reachability problem
        """
        # Decision variables - value function
        V = cp.Variable(self.S)

        # Minimize value function
        objective = cp.Minimize(cp.sum(V))

        # Bellman constraints
        constraints = []
        for s in range(self.S):
            constraints.append(V[s] <= 1.0)
            constraints.append(V[s] >= 0.0)
            (mdp_s, q) = self.states[s]
            if (mdp_s, q) in self.acc:
                constraints.append(V[s] == 1.0)
            elif (mdp_s, q) in self.sink:
                constraints.append(V[s] == 0.0)
            else:
                for a in range(self.A):
                    future_rew = self.P[a][s, :] @ V
                    constraints.append(V[s] >= future_rew)

        # Optimization problem
        prob = cp.Problem(objective, constraints)

        # Solve
        prob.solve()
        V_star = V.value

        # Extract policy
        policy = self.extract_policy(V_star)

        return V_star, policy


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
                s_new, _ = self.sample_model(s_batch[i], a_batch[i])
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