import numpy as np
from mdp import MDP
from dfa import DFA
from product_mdp import ProductMDP

ROW = 4
COL = 5
S = ROW*COL
A = 4
stay_prob = 0.8
slip_prob = (1 - stay_prob)/2

P = np.zeros((A, S, S))

def idx(row, col):
    return row * COL + col

# movement directions
action_delta = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}

barrier_cells = [(0,4), (1,0), (1,2), (1,4), (2,0), (2,2), (2,4), (3,0), (3,2), (3,4)]
gray_cells = [(1,1), (2,1), (3,1), (1,3), (2,3), (3,3)]

barrier_set = {idx(r, c) for (r, c) in barrier_cells}
gray_set = {idx(r, c) for (r, c) in gray_cells}

for row in range(ROW):
    for col in range(COL):
        s = idx(row, col)

        if s in barrier_set:
            P[:, s, s] = np.ones_like(P[:, s, s])
            continue
    
        for a in range(A):
            if s in gray_set and a in [0, 1]:
                slip_left = (-1, -1) if a == 0 else (1, -1)  # slip mapping
                slip_right = (-1, 1) if a == 0 else (1, 1)
                moves = [
                    (action_delta[a], stay_prob), # vertical move
                    (slip_left, slip_prob),       # slip left
                    (slip_right, slip_prob)       # slip right
                ]

                for (drow, dcol), p in moves:
                    new_row, new_col = row + drow, col + dcol
                    if 0 <= new_row < ROW and 0 <= new_col < COL:
                        new_s = idx(new_row, new_col)
                        P[a, s, new_s] = p
                    else:
                        P[a, s, s] += p

            else: # deterministic
                drow, dcol = action_delta[a]
                new_row, new_col = row + drow, col + dcol

                if 0 <= new_row < ROW and 0 <= new_col < COL:
                    new_s = idx(new_row, new_col)
                    P[a, s, new_s] = 1.0
                else:
                    P[a, s, s] = 1.0

# Check for correct probability matrix
# print(P.shape)
# print(P[3].sum(axis=1))

grid_world = MDP(0, P)

mona_dfa_string1 = """
digraph MONA_DFA {
rankdir = LR;
center = true;
size = "7.5,10.5";
edge [fontname = Courier];
node [height = .5, width = .5];
node [shape = doublecircle]; 3;
node [shape = circle]; 1;
init [shape = plaintext, label = ""];
init -> 1;
1 -> 1 [label="~a"];
1 -> 2 [label="a & ~b"];
1 -> 3 [label="a & b"];
2 -> 2 [label="~b"];
2 -> 3 [label="b"];
3 -> 3 [label="true"];
}
"""

mona_dfa_string2 = """
digraph MONA_DFA {
rankdir = LR;
center = true;
size = "7.5,10.5";
edge [fontname = Courier];
node [height = .5, width = .5];
node [shape = doublecircle]; 2;
node [shape = circle]; 1;
init [shape = plaintext, label = ""];
init -> 1;
1 -> 1 [label="~c"];
1 -> 2 [label="c"];
2 -> 2 [label="true"];
}
"""

dfa1 = DFA(mona_dfa_string1)
dfa2 = DFA(mona_dfa_string2)

# print(dfa.states)
# print(dfa.aps)
# print(dfa.labels)
# print(dfa.q0)
# print(dfa.acc)
# print(dfa.shape)
# print(dfa.delta[1])

# Labeling function: L(s) -> list of s dictionaries, where each is the "context" for the atomic proposition
A_idx, B_idx, C_idx = idx(0, 1), idx(3, 1), idx(3, 3)

labels1 = [{'a': False, 'b': False} for s in range(grid_world.S)]
labels1[A_idx]['a'] = True
labels1[B_idx]['b'] = True

labels2= [{'c': False} for s in range(grid_world.S)]
labels2[C_idx]['c'] = True

env1 = ProductMDP(dfa1, grid_world, labels=labels1)
env2 = ProductMDP(dfa2, grid_world, labels=labels2)

# print(env.final)
# print(env.alphabet)
# print(env.rewards)
# print([env.P[i].shape for i in range(env.A)])
# print([env.P[i].sum(axis=1).sum() for i in range(env.A)])
# print(env.labels)
# print(env.P[2].sum(axis=1))
# print(env.acc)

"""
Value Iteration
"""
# V1, Q1, policy1 = env1.value_iteration()

# states_idx_before_A = [env1.states.index((s, 0)) for s in range(grid_world.S)]
# states_idx_after_A = [env1.states.index((s, 1)) for s in range(grid_world.S)]

# V_before_A = V1[states_idx_before_A].reshape((ROW, COL))
# V_after_A = V1[states_idx_after_A].reshape((ROW, COL))

# print(V_before_A)
# print(V_after_A)

# policy_before_A = policy1[states_idx_before_A].reshape((ROW, COL))
# policy_after_A = policy1[states_idx_after_A].reshape((ROW, COL))

# print(policy_before_A)
# print(policy_after_A)

V2, Q2, policy2 = env2.value_iteration()
states_idx = [env2.states.index((s, 0)) for s in range(grid_world.S)]

V_states = V2[states_idx].reshape((ROW, COL))
policy_states = policy2[states_idx].reshape((ROW, COL))

print(V_states)
print(policy_states)