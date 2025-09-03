import numpy as np
x = [1, 2]
tst = np.array([x for i in range(2)])
# print(tst[:, 0])

def tstMap(a, b):
    return [a, b]

a = [1, 2]
b = [3, 4]
tst2 = np.array(tuple(map(tstMap, a, b)))
# print(tst2)

def tstMatMap(m1, m2):
    return np.multiply(m1, m2)

ma = np.array([
    [[1, 1],
    [2, 2]],
    [[1, 1],
     [2, 2]]
])

mb = np.array([
    [[2, 2],
    [2, 2]],
    [[3, 2],
     [2, 3]]
])

tst3 = tuple(map(tstMatMap, ma, mb))
# print(tst3[0])

ta = tuple(ma[i] for i in range(2))
tb = tuple(mb[i] for i in range(2))
tst4 = np.array(tuple(map(tstMatMap, ta, tb)))
# print(tst4[0])

a = []
for i in range(3):
    a += [(i, s) for s in range(2)]
# print(a)

map = [(s, q) for s in range(4) for q in range(3)]
# print(map)

V = np.array([1, 2, 3, 4, 6])
print(V[[0, 2, 4]])