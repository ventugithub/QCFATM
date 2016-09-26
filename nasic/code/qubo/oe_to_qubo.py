import copy
import itertools as it
import yaml

import numpy as np

def qubo_sum(Q, QQ):
    qubo = copy.deepcopy(Q)
    for ij, v in QQ.items():
        qubo[ij] = qubo.get(ij, 0) + v
    return qubo

def qubo_product(Q, QQ):
    qubo = {}
    for i, v in Q:
        qubo[i] = v ** 2
    for i, v in QQ:
        qubo[i] = qubo.get(i, 0) + (v ** 2)
    for (i, v), (ii, vv) in it.product(Q.items(), QQ.items()):
        j = tuple(sorted(set(i + ii)))
        qubo[j] = qubo.get(j, 0) + (2 * v * vv)
    return qubo

# parse instance file
instance_filepath = 'example_1.oe'
with open(instance_filepath) as instance_file:
    instance_parameters = yaml.load(instance_file)
I = instance_parameters['I']
J = instance_parameters['J']
L = instance_parameters['L']
assert np.array(instance_parameters['b'], dtype=int).shape == (I, L)
assert np.array(instance_parameters['M'], dtype=int).shape == (L, J)
assert np.array(instance_parameters['r'], dtype=int).shape == (L, J, L)
b = lambda i, l: instance_parameters['b'][i][l]
M = lambda l, j: float(instance_parameters['M'][l][j])
r = lambda l, j, ll: float(instance_parameters['r'][l][j][ll])

# defaults
A = lambda l: sum([b(i, l) for i in range(I)])
B = lambda i: sum([b(i, l) for l in range(L)])
K1 = lambda i, j: range(L)
N = lambda i, j: range(I)
Q = lambda i, j, l, ii: range(L)
K2 = lambda i, j, l: [(ii, ll) for ii in N(i, j) for ll in Q(i, j, l, ii)]

IJ = list(it.product(*map(range(I, J))))

hamiltonian_terms = ('airfield-type', 'airfield', 'type',
                     'task', 'cover', 'cost')
penalty_weights = {
    {'airfield-type': 1,
     'airfield': 1,
     'type': 1,
     'task': 1,
     'cover': 1,
     'cost': 1
}

# parse instance file
instance_filepath = 'example_1.oe'
with open(instance_filepath) as instance_file:
    instance_parameters = yaml.load(instance_file)
I = instance_parameters['I']
J = instance_parameters['J']
L = instance_parameters['L']
assert np.array(instance_parameters['M'], dtype=int).shape == (L, J)
assert np.array(instance_parameters['b'], dtype=int).shape == (I, L)
assert np.array(instance_parameters['r'], dtype=int).shape == (L, J, L)
M = lambda l, j: instance_parameters['M'][l][j]
b = lambda i, l: instance_parameters['b'][i][l]
r = lambda l, j, ll: instance_parameters['r'][l][j][ll]

# defaults
A = lambda l: sum([b(i, l) for i in range(I)])
B = lambda i: sum([b(i, l) for l in range(L)])
K1 = lambda i, j: range(L)
N = lambda i, j: range(I)
Q = lambda i, j, l, ii: range(L)
K2 = lambda i, j, l: [(ii, ll) for ii in N(i, j) for ll in Q(i, j, l, ii)]

IJ = list(it.product(*map(range(I, J))))
IJL = [(i, j, l) for i, j in IJ for l in K1(i, j)]

# set up indexing
# ('x', (i, j, l), k)
# ('y', (i, j, l), (ii, ll), k)
num_qubits = 0
from_qubo_index = {}
x, y, z = {}, {}, {}
for i, j, l in IJL:
    for k in b(i, l):
        x[(i, j, l), k] = num_qubits
        from_qubo_index[num_qubits] = ('x', (i, j, l), k)
        num_qubits += 1
    for ii, ll in K2(i, j, l):
        for k in b(ii, ll):
            y[(i, j, l), (ii, ll), k] = num_qubits
            from_qubo_index[num_qubits] = ('y', (i, j, l), (ii, ll), k)
            num_qubits += 1

## Hamiltonian

subqubos = {}
# force strength (airfield, type)
# for all i, l:
# sum_j x_{i, j, l} + sum_{ii, jj, ll} y_{ii, jj, ll, i, l} <= b(i, l)
# ('airfield-type', (i, l), k)
# X + Y <= b >>> (X + Y + Z - b)^2
# X + Y + Z = b >>> Z <= b
subqubo = {}
for i, l in it.product(*map(range, (I, L))):
    X = {}
    for j in range(J):
        if (i, j, l) in IJL:
            for k in range(b(i, l)):
                X[x[(i, j, l), k]] = 1
    Y = {}
    for ii, jj, ll in IJL:
        if (i, l) in K2(ii, jj, ll):
            for k in range(b(i, l)):
                Y[y[(ii, jj, ll), (i, l), k]] = 1
    Z = {}
    for k in range(b(i, l)):
        Z['airfield-type', (i, l), k] = num_qubits
        from_qubo_index[num_qubits] = ('airfield-type', (i, l), k)
        Z[num_qubits] = 1
        num_qubits += 1
    C = {(,): -b(i, l)}
    S = reduce(qubo_sum, (X, Y, Z, C))
    P = qubo_product(S, S)
    for i in P:
        P[i] *= penalty_weight['airfield-type']
    subqubo = qubo_sum(sub_qubo, P)
subqubos['airfield-type'] = subqubo = {}

# force strength (airfield)
# for all i:
# sum_{j, l} x_{i, j, l} + sum_{ii, jj, ll, l} y_{ii, jj, ll, i, l} <= B(i)
# ('airfield', i, k)
if 'B' in instance_parameters:
    assert 0, 'Assumed B(i) = sum_l b(i, l).'
else:
    subqubo = {}
subqubos['airfield'] = subqubo

# force strength (type)
# for all l:
# sum_{i, j} x_{i, j, l} + sum_{ii, jj, ll, i} y_{ii, jj, ll, i, l} <= A(l)
# ('type', l, k)
if 'A' in instance_parameters:
    assert 0, 'Assumed A(l) = sum_i b(i, l).'
else:
    subqubo = {}
subqubos['type'] = subqubo

# task completion
# for all j:
# sum_{i, l} x_{i, j, l} / M(l, j) >= 1
# ('task', j, k)
# X >= 1 >>> (X - Z - 1)^2
# (Z = X - 1) && (X <= b) >>> Z <= b - 1
subqubo = {}
for j in range(J):
    X = {}
    for i, l in it.product(*map(range, (I, L))):
        if (i, j, l) in IJL:
            for k in range(b(i, l)):
                X[x[(i, j, l), k]] = 1
    Z = {}
    # if l not N(i, j) for some i, then this range can be reduced
    for k in range(int(math.ceil(sum([A(l) / M(l, j) for l in range(L)]))):
        Z['task', j, k] = num_qubits
        from_qubo_index[num_qubits] = 'Z', j, k
        Z[num_qubits] = -1
        num_qubits += 1
    C = {(,): -1}
    S = reduce(qubo_sum, (X, Y, C))
    P = qubo_product(S, S)
    for i in P:
        P[i] *= penalty_weight['task']
    subqubo = qubo_sum(sub_qubo, P)
subqubos['task'] = subqubo

# escort cover
# for all i, j, l:
# sum_{ii, ll} y_{i,j,l,ii,ll} / r(l, j, ll) >= x_{i, j, l}
# ('cover', (i, j, l), k)
# Y >= X >>> (Y - X - Z)^2
# Z = Y - X <= Y
subqubo = {}
for i, j, l in IJL:
    Y = {}
    for ii, ll in K2(i, j, l):
        for k in range(b(ii, ll)):
            Y[y[(i, j, l), (ii, ll), k]] = 1
    X = {}
    for k in range(b(i, l)):
        X[x[(i, j, l), k]] = -1
    Z = {}
    for k in range(int(math.ceil(sum([b(ii, ll) / r(i, j, ll)
                                      for ii, ll in K2(i, j, l)])))):
        Z['cover', (i, j, l), k] = num_qubits
        from_qubo_index[num_qubits] = 'cover', (i, j, l), k
        Z[num_qubits] = -1
        num_qubits += 1
    S = qubo_sum(Y, X, Z)
    P = qubo_product(S, S)
    for i in P:
        P[i] *= penalty_weight['cover']
    subqubo = qubo_sum(sub_qubo, P)
subqubos['cover'] = subqubo

# cost function
subqubos['cost'] = {}

# total Hamiltonian
qubo = reduce(qubo_sum, sub_qubos.values())
