from dwave_sapi2.embedding import find_embedding
from dwave_sapi2.util import get_hardware_adjacency
from dwave_sapi2.local import local_connection

import itertools as it
import numpy as np
import random

def randomConnection(N=10, P=0.5):
    J = {}
    for i, j in it.combinations(np.arange(N), 2):
        r = random.random()
        if r < P:
            J[(i, j)] = 1
    return J

def main():
    solver = local_connection.get_solver("c4-sw_sample")
    A = get_hardware_adjacency(solver)
    l = []
    for N in range(10, 11):
        for P in np.arange(0.0, 0.1, 0.001):
            J = randomConnection(N, P)
            embedding = find_embedding(J, A, fast_embedding=True)
            l.append(len(embedding))
    return l

if __name__ == "__main__":
    l = main()
    import matplotlib.pyplot as plt
    plt.plot(l)
    plt.show()
