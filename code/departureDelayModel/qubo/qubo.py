from polynomial import Polynomial as poly
import instance
import variable

def isRealConflict(delay1, delay2, deltaTMin, deltaTMax, threshold=3):
    dmin = - threshold - deltaTMax
    dmax = threshold - deltaTMin
    delayDiff = delay1 - delay2
    if delayDiff > dmin and delayDiff < dmax:
        return 1
    else:
        return 0

def get_number_of_logical_qubits(input):
    """ Read in instance and calculate the QUBO as well as the index mapping """
    inst = instance.Instance(input)

    flights = inst.flights
    delays = inst.delays
    I = len(flights)
    D = len(delays)
    return D * I

def get_qubo(input, penalty_weights, unary=False):
    """ Read in instance and calculate the QUBO as well as the index mapping """
    inst = instance.Instance(input)

    flights = inst.flights
    delays = inst.delays
    conflicts = inst.conflicts
    timeLimits = inst.timeLimits
    I = len(flights)
    K = len(conflicts)

    delayValues = list(delays)

    var = variable.Unary(inst)

    NDelay = var.NDelay

    penalty_weights['departure'] = 1.0/delayValues[-1]
    if not unary:
        raise ValueError('Binary representation is not feasible for this model due to the conflict penalizing term in the cost function')

    ###########################################################################
    # calculate QUBO
    ###########################################################################
    qubo = poly()
    subqubos = {}
    print "Calculate departure delay contribution"
    subqubos['departure'] = poly()
    for i in range(I):
        subqubos['departure'] += var.delay(i)
    qubo += penalty_weights['departure'] * subqubos['departure']

    print "Calculate conflict contribution"
    subqubos['conflict'] = poly()
    flights = var.instance.flights
    for k in range(K):
        f1, f2 = conflicts[k]
        i = flights.index(f1)
        j = flights.index(f2)
        Q = poly()
        for a in range(NDelay):
            for b in range(NDelay):
                if isRealConflict(delayValues[a], delayValues[b], timeLimits[k][0], timeLimits[k][1]):
                    Q += poly({(var.d[i, a],): 1}) * poly({(var.d[j, b],): 1})
        subqubos['conflict'] += Q
    qubo += penalty_weights['conflict'] * subqubos['conflict']

    print "Calculate departure delay uniqueness contribution"
    subqubos['unique'] = poly()
    for i in range(I):
        Q = poly()
        for a in range(NDelay):
            Q += poly({(var.d[i, a],): 1})
        Q += poly({(): -1})
        subqubos['unique'] += Q * Q
    qubo += penalty_weights['unique'] * subqubos['unique']

    if not qubo.isQUBO():
        print "WARNING: Cost function is not quadratic!"

    return qubo, subqubos, var
