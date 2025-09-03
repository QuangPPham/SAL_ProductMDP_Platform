"""
Class for specifying a grid world as an MDP

Parameters
    ----------
    size : tuple
        Specifies width and height of grid
    reward : array
        Reward matrices or vectors. Like the transition matrices, these 
        can also be defined in a variety of ways. The simplest are numpy
        arrays with shape (S, A), or (S,), or (A, S, S).
    discount : float
        Discount factor ∈ (0, 1]. The per time-step discount factor on future rewards.
    epsilon : float
        Stopping criterion for value iteration, which is the base method for solving MDP.

    Attributes
    ----------
    P : array
        Transition probability matrices.
    R : array
        Reward vectors.
    V : tuple
        The optimal value function. Each element is a float corresponding to
        the expected value of being in that state assuming the optimal policy
        is followed.
    Q: array
        The optimal action-value function. Each element corresponds to the expected
        value of being in that state, taking that action, then following the optimal
        policy forever after.
    policy : tuple
        The optimal policy.

    Methods
    -------
    runVI()
        Implement value iteration.
    sample(s, a)
        Sample next state s', and reward r from transition probabilities.
        Used to test model-free algorithms such as MCTS or Q-learning.
"""