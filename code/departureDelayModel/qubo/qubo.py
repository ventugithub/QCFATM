import progressbar
import numpy as np

from polynomial import Polynomial as poly
import instance
import variable

conflictPenalty = 5
threshold = 3
def penalizeConflict(delta, t1, t2, threshold):
    diff = abs(delta + t1 - t2)
    if diff < threshold:
        return 1
    else:
        return 0

def get_qubo(input, unary=False):
    """ Read in instance and calculate the QUBO as well as the index mapping """
    inst = instance.Instance(input)

    flights = inst.flights
    delays = inst.delays
    conflicts = inst.conflicts
    arrivalTimes = inst.arrivalTimes
    I = len(flights)
    K = len(conflicts)

    delayValues = list(delays)
    deltaValues = np.concatenate((np.sort(-np.array(inst.delays[1:])), np.array(inst.delays))).tolist()

    if unary:
        var = variable.Unary(inst)
    else:
        var = variable.Binary(inst)

    NDelay = var.NDelay
    NDelta = var.NDelta

    penalty_weights = {
        'departure': 1.0 / delayValues[1],
        'conflict': 1,
        'boundary-condition': 1.0 / pow(delayValues[1], 2),
        'departure-unique': 1,
        'conflict-unique': 1
    }
    if not unary:
        raise ValueError('Binary representation is not feasible for this model due to the conflict penalizing term in the cost function')

    ###########################################################################
    # calculate QUBO
    ###########################################################################
    qubo = poly()
    subqubos = {}
    print "Calculate departure delay contribution"
    pbar = progressbar.ProgressBar().start()
    pbar.maxval = I
    count = 0
    subqubos['departure'] = poly()
    for i in range(I):
        subqubos['departure'] += var.delay(i)
        pbar.update(count)
        count = count + 1
    qubo += penalty_weights['departure'] * subqubos['departure']
    pbar.finish()

    print "Calculate conflict contribution"
    pbar = progressbar.ProgressBar().start()
    pbar.maxval = K
    count = 0
    subqubos['conflict'] = poly()
    for k in range(K):
        for a in range(NDelta):
            penalty = penalizeConflict(deltaValues[a], arrivalTimes[k][0], arrivalTimes[k][1], threshold)
            subqubos['conflict'] += poly({(var.D[k, a],): penalty})
        pbar.update(count)
        count = count + 1
    qubo += penalty_weights['conflict'] * subqubos['conflict']
    pbar.finish()

    print "Calculate boundary condition contribution"
    pbar = progressbar.ProgressBar().start()
    pbar.maxval = K
    count = 0
    subqubos['boundary-condition'] = poly()
    for k in range(K):
        Q = poly()
        Q += var.delta(k)
        i, j = conflicts[k]
        Q -= var.delay(i)
        Q += var.delay(j)
        subqubos['boundary-condition'] = Q * Q
        pbar.update(count)
        count = count + 1
    qubo += penalty_weights['boundary-condition'] * subqubos['boundary-condition']
    pbar.finish()

    if unary:
        print "Calculate departure delay uniqueness contribution"
        pbar = progressbar.ProgressBar().start()
        pbar.maxval = I
        count = 0
        subqubos['departure-unique'] = poly()
        for i in range(I):
            Q = poly()
            for a in range(NDelay):
                Q += poly({(var.d[i, a],): 1})
            Q += poly({(): -1})
            subqubos['departure-unique'] += Q * Q
            pbar.update(count)
            count = count + 1
        qubo += penalty_weights['departure-unique'] * subqubos['departure-unique']
        pbar.finish()

        print "Calculate arrival time difference uniqueness contribution"
        pbar = progressbar.ProgressBar().start()
        pbar.maxval = K
        count = 0
        subqubos['conflict-unique'] = poly()
        for k in range(K):
            Q = poly()
            for a in range(NDelta):
                Q += poly({(var.D[k, a],): 1})
            Q += poly({(): -1})
            subqubos['conflict-unique'] += Q * Q
            pbar.update(count)
            count = count + 1
        qubo += penalty_weights['conflict-unique'] * subqubos['conflict-unique']
        pbar.finish()

    # save qubos
    print "Save QUBO ..."
    filename = "%s.qubo.yaml" % input
    qubo.save(filename)
    if not qubo.isQUBO():
        print "WARNING: Cost function is not quadratic!"

    return qubo, subqubos, var
