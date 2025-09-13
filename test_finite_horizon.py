import numpy as np
from mdp import MDP
from Q_Learning import QLearning

A = 2 # 0: left, 1: right
S = 4

P = np.zeros((A, S, S))

for s in range(S):
    for snew in range(S):
        if snew == s + 1:   # next state is to the right
            P[0, s, snew] = 0.2
            P[1, s, snew] = 0.9
        elif snew == s - 1: # next state is to the left
            P[0, s, snew] = 0.8
            P[1, s, snew] = 0.1
        if s == 0:
            P[0, s, s] = 0.8 # going left stays in the same state
            P[1, s, s] = 0.1 # 0.1 chance of going left when going right
        elif s == 3:
            P[0, s, s] = 0.2 # 0.2 chance of going right when going left
            P[1, s, s] = 0.9 # going right stays in the same state

R = np.zeros((A, S, S)) # R(s, a, s')

for a in range(A):
    for s in range(S):
        for snew in range(S):
            if snew == 0:
                rew = 10.
            elif snew == 1 or snew == 2:
                rew = 0.5
            elif snew == 3:
                rew = 3.
            R[a, s, snew] = rew


mdp = MDP(transitions=P, rewards=R, horizon=2, discount=1.0)

print("============Value Iteration=============")
V, Q, policy = mdp.value_iteration()
for i in range(mdp.T):
    print(f"At timestep {i}")
    print(f"Value: {V[i]}")
    print(f"Policy: {policy[i]}")

print("============Policy Iteration=============")
V_PI, policy_PI = mdp.policy_iteration(iterative=True)
for i in range(mdp.T):
    print(f"At timestep {i}")
    print(f"Value: {V_PI[i]}")
    print(f"Policy: {policy_PI[i]}")