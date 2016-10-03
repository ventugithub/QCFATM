import progressbar

from polynomial import Polynomial as poly
import instance
import variable

def isRealConflict(delay1, delay2, t1, t2, threshold=3):
    diff = abs(delay1 - delay2 + t1 - t2)
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

    var = variable.Unary(inst)

    NDelay = var.NDelay

    penalty_weights = {
        'departure': 1.0,
        'conflict': delayValues[-1] * 1.0,
        'unique': delayValues[-1] * 1.0,
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
        i, j = conflicts[k]
        Q = poly()
        for a in range(NDelay):
            for b in range(NDelay):
                if isRealConflict(delayValues[a], delayValues[b], arrivalTimes[k][0], arrivalTimes[k][1]):
                    Q += poly({(var.d[i, a],): 1}) * poly({(var.d[j, b],): 1})
        subqubos['conflict'] += Q
        pbar.update(count)
        count = count + 1
    qubo += penalty_weights['conflict'] * subqubos['conflict']
    pbar.finish()

    print "Calculate departure delay uniqueness contribution"
    pbar = progressbar.ProgressBar().start()
    pbar.maxval = I
    count = 0
    subqubos['unique'] = poly()
    for i in range(I):
        Q = poly()
        for a in range(NDelay):
            Q += poly({(var.d[i, a],): 1})
        Q += poly({(): -1})
        subqubos['unique'] += Q * Q
        pbar.update(count)
        count = count + 1
    qubo += penalty_weights['unique'] * subqubos['unique']
    pbar.finish()

    if not qubo.isQUBO():
        print "WARNING: Cost function is not quadratic!"

    return qubo, subqubos, var
