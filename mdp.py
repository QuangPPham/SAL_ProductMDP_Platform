"""
Class for specifying a Markov Decision Process.

    Let ``S`` = the number of states, and ``A`` = the number of acions.

    Parameters
    ----------
    transitions : array
        Transition probability matrices. Numpy array with shape (A, S, S):
        a tuple or list or numpy object array of length A, where each 
        element contains a numpy matrix with shape (S, S`). In summary, each
        action's transition matrix must be indexable like transitions[a]
        where a ∈ {0, 1...A-1}, and transitions[a] returns an SxS array-like object.
    rewards : array
        Reward matrices or vectors. Like the transition matrices, these 
        can also be defined in a variety of ways. The simplest are numpy
        arrays with shape (S, A), or (S,), or (A, S, S).

    If transition probabilities and expected rewards are not known apriori, then specify sample_model
    sample_model : function
        take in a state-action pair, then compute the next state and reward based on some incremental dynamics
        
    discount : float
        Discount factor ∈ (0, 1]. The per time-step discount factor on future rewards.
    epsilon : float
        Stopping criterion for value iteration, which is the base method for solving MDP.

    Attributes
    ----------
    P : tuple of arrays
        Transition probability matrices.
    R : tuple of vectors
        Reward vectors.

    Methods
    -------
    Bellman_update(Vprev)
        One step of the value iteration update using either Jacobi or Gauss-Seidel method.
        Returns value and action-value functions, and improved epsilon-greedy policy.
    value_iteration()
        Performs value iteration using the Bellman update, until a threshold of epsilon
    computePR_policy(policy)
        Computes transition matrix (s,s') and expected reward (s') for a given policy
        If no policy passed, returns transition and reward for optimal policy
    eval(policy)
        Evaluate the value function of a given policy analytically
    get_action(policy)
        Returns an action for state s according to the given policy.
        Supports both deterministic (1D) and stochastic (2D) policies.
    rollout(s, a, policy, depth)
        Sample next state s', and reward r from transition probabilities,
        If depth > 1, sample action from policy then continue sampling
        Returns trajectory: s[1 -> depth], a[1 -> depth-1], r[1 -> depth-1]
    sample(s, a)
        Sample next state directly

"""
import numpy as np

def _computeDimensions(transition):
    A = len(transition)
    try:
        if transition.ndim == 3:
            S = transition.shape[1]
        else:
            S = transition[0].shape[0]
    except AttributeError: 
        # in case transition does not have ndim attribute
        S = transition[0].shape[0]
    
    return S, A

def _computeRewards(transition, reward, S, A):
    def _computeMatrixReward(rewards, transitions):
        # r(s,a) = E(s')[r(s,a,s')] = \sum_s' [P(s'|s,a)*R(s,a,s')]
        return np.multiply(transitions, rewards).sum(1).reshape(S) # Make sure reward[a] has shape (s,)
    
    # make sure rewards can be indexed as reward[a]
    try:
        # (s,) -> r(s')
        if reward.ndim == 1:
            r = np.array(reward).reshape(S)     # Make sure the shape is (s,)
            return tuple([r for a in range(A)]) # the reward from this state is the same no matter the action (?)
        
        # (s, a) -> r(s,a)
        elif reward.ndim == 2:
            def func(x):
                return np.array(x).reshape(S)                    # Make sure shape is (s,)
            return tuple([func(reward[:, a]) for a in range(A)])
        
        # (s, a, s) -> r(s, a, s') -> r(s,a)
        else:
            # map() applies function to every iterable in reward and function
            r = tuple(map(_computeMatrixReward, reward, transition))
            return r  # return a-length-tuple of s-length np arrays
        
    # If reward is not numpy array
    except (AttributeError, ValueError):
        if len(reward) == A:
            r = tuple(map(_computeMatrixReward, reward, transition))
            return r
        else:
            r = np.array(reward).reshape(S)
            return tuple([r for a in range(A)])

class MDP():
    def __init__(self, s0 = 0, transitions=None, sample_model=None, rewards=None, discount=1.0, epsilon=1e-6, max_iter=10000):
        # Initial state
        self.s0 = s0

        # Discount factor
        if discount is not None:
            self.gamma = float(discount)
            assert 0.0 < self.gamma <= 1.0, "Discount must be in (0, 1]"
        
        # Maximum iterations
        if max_iter is not None:
            self.max_iter = max_iter
            assert self.max_iter > 0, "Max iterations must be greater than 0."

        # Epsilon
        if epsilon is not None:
            self.epsilon = float(epsilon)
            assert self.epsilon > 0, "Epsilon must be greater than 0."

        # MDP dynamics
        if transitions is not None:
            self.S, self.A = _computeDimensions(transitions)
            self.P = tuple([transitions[a] for a in range(self.A)])        # length-A tuple of SxS np arrays -> containing P(s' | s,a)
        else:
            self.P = None
            self.sample_model = sample_model

        if rewards is not None:
            self.R = _computeRewards(transitions, rewards, self.S, self.A) # length-A tuple of length-S np vector -> containng R(s, a)
        else:
            self.R = None

        # User-implemented
        self.V = None            # (S,) vector containing V(s)
        self.Q = None            # (A, S) matrix containing Q(s, a)
        self.policy = np.random.randint(self.A, size=self.S)
                                # (S,) vector containing deterministic action indices or
                                # (S, A) array contraining probability of action a at state s

    def Bellman_update(self, Vprev, GS=True):
        """
        One step of the value iteration update using either Jacobi or Gauss-Seidel method.
        Returns value and action-value functions, and improved epsilon-greedy policy
        """

        try:
            assert Vprev.shape in ((self.S,), (1, self.S)), "V is not the right shape (Bellman operator)."
        except AttributeError:
            raise TypeError("V must be a numpy array or matrix.")

        Q = np.empty((self.A, self.S))

        if GS:
            V = Vprev.copy()
            policy = np.empty_like(Vprev)
            for s in range(self.S):
                for a in range(self.A):
                    Q[a, s] = self.R[a][s] + self.gamma * self.P[a][s, :].dot(V)
                V[s] = Q[:,s].max()
                policy[s] = Q[:, s].argmax()
        else:
            for a in range(self.A):
                Q[a] = self.R[a] + self.gamma * self.P[a].dot(Vprev)

            V = Q.max(axis=0)
            policy = Q.argmax(axis=0)

        return V, Q, policy
    
    def value_iteration(self, GS=True):
        V = np.zeros(self.S)
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
        
        P_pi = np.empty((self.S, self.S))
        R_pi = np.empty(self.S) # R(s') given s' from s,a

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
            for s_new in range(self.S):
                for s in range(self.S):
                    P_pi[s, s_new] = policy[s, :].dot(P[:, s, s_new])
                    for a in range(self.A):
                        R_pi[s_new] += policy[s, a]*self.P[a][s, s_new]*self.R[a][s]
        else:
            raise ValueError("Policy must be 1D (deterministic) or 2D (stochastic)")

        return P_pi, R_pi    

    def eval(self, policy=None):
        """
        Evaluate value function of a given policy
        Using analytical approach
        """
        if policy is None:
            return self.V
        
        # V = PR + gPV  => (I-gP)V = PR  => V = inv(I-gP)*PR
        P_pi, R_pi = self.computePR_policy(policy)
        V_pi = np.linalg.solve((np.eye(self.S) - self.gamma*P_pi), R_pi)

        Q_pi = np.empty((self.A, self.S))
        for a in range(self.A):
            Q_pi[a, :] = self.R[a][:] + self.gamma * self.P[a].dot(V_pi)

        return V_pi, Q_pi

    def policy_evaluation_iterative(self, policy, max_iter = 1000, GS=True):
        """
        Iteratively evaluate the value function for a given policy.
        policy: array of shape (SX,) for deterministic, or (SX, A) for stochastic
        Returns: V_pi (value function under policy)
        """

        V = np.zeros(self.S)
        for _ in range(max_iter):
            V_prev = V.copy()
            if policy.ndim == 1:  # deterministic
                for s in range(self.S):
                    a = int(policy[s])
                    if GS:
                        V[s] = self.R[a][s] + self.gamma * self.P[a][s, :].dot(V)
                    else:
                        V[s] = self.R[a][s] + self.gamma * self.P[a][s, :].dot(V_prev)
            elif policy.ndim == 2:  # stochastic
                for s in range(self.S):
                    for a in range(self.A):
                        if GS:
                            V[s] += policy[s, a] * (self.R[a][s] + self.gamma * self.P[a][s, :].dot(V))
                        else:
                            V[s] += policy[s, a] * (self.R[a][s] + self.gamma * self.P[a][s, :].dot(V_prev))
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

        q_values = np.empty((self.A, self.S))
        for a in range(self.A):
            q_values[a] = self.R[a] + self.gamma * self.P[a].dot(V)
        policy = q_values.argmax(axis=0)
   
        return policy
    
    def policy_iteration(self, iterative = True, GS = True):
        V = np.zeros(self.S)
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

    def get_action(self, s, policy=None):
        """
        Returns an action for state s according to the given policy.
        Supports both deterministic (1D) and stochastic (2D) policies.
        """
        if policy is None:
            policy = self.policy

        if policy.ndim == 1:
            # policy is deterministic
            a_new = policy[s]
        elif policy.ndim == 2:
            # policy is stochastic
            a_new = np.random.choice(self.A, p=policy[s])
        else:
            raise ValueError("Policy must be 1D (deterministic) or 2D (stochastic)")
        
        return a_new

    def rollout(self, s, a=None, policy=None, depth=1):
        """
        Simulate the trajectory of the MDP. If action is not given, select action
        according to policy. If depth is greater than 1, select future actions based
        on the policy. Returns the array of states, actions, and rewards. 

        Note that terminal state is included, but terminal action and reward are not.
        """
        if a is None:
            a = self.get_action(s, policy)
            
        s_batch, a_batch, r_batch = [s], [a], [self.R[a][s]]

        for i in range(depth):
            if self.P is not None:
                probs = self.P[a_batch[i]][s_batch[i], :] # get distribution over s' given s and a
                s_new = np.random.choice(self.S, p=probs)
                a_new = self.get_action(s_new, policy)
                r_new = self.R[a_new][s_new]
            else:
                s_new, r_new = self.sample_model(s_batch[i], a_batch[i])
                a_new = self.get_action(s_new, policy)

            s_batch += [s_new]
            a_batch += [a_new]
            r_batch += [r_new]

        return s_batch, a_batch[:-1], r_batch[:-1]
    
    def sample(self, s, a):
        # Sample next state and reward
        s_list, a_list, r_list = self.rollout(s, a, depth=1)
        return s_list[-1], r_list[-1]