import qcfco.qubo
from qcfco.polynomial import Polynomial as poly


def isRealConflict(delay1, delay2, deltaTMin, deltaTMax, threshold=3):
    dmin = - threshold - deltaTMax
    dmax = threshold - deltaTMin
    delayDiff = delay1 - delay2
    if delayDiff > dmin and delayDiff < dmax:
        return 1
    else:
        return 0


class Qubo(qcfco.qubo.Qubo):
    def __init__(self, instance, variable, penalty_weights):
        """ calculate the QUBO

        instance: instance class instance
        variable: variable class instance
        penalty_weights: list of penalty weights

        """
        self.instance = instance
        self.variable = variable
        self.penalty_weights = penalty_weights

    @staticmethod
    def get_hard_constraints():
        """ return a list of subqubo names whose value must be zero to fulfill hard constraints"""
        return ['unique', 'conflict']

    @staticmethod
    def get_subqubo_names():
        """ return a list of subqubo names"""
        return ['departure', 'unique', 'conflict']

    def get_num_logical_qubits(self):
        flights = self.instance.flights
        delays = self.instance.delays
        I = len(flights)
        D = len(delays)
        return D * I

    def get_qubo(self):
        ###########################################################################
        # calculate QUBO
        ###########################################################################

        flights = self.instance.flights
        delays = self.instance.delays
        conflicts = self.instance.conflicts
        timeLimits = self.instance.timeLimits
        I = len(flights)
        K = len(conflicts)

        delayValues = list(delays)

        var = self.variable

        NDelay = var.NDelay

        ###########################################################################
        # calculate QUBO
        ###########################################################################
        qubo = poly()
        subqubos = {}
        print "Calculate departure delay contribution"
        subqubos['departure'] = poly()
        for i in range(I):
            subqubos['departure'] += var.delay(i)
        subqubos['departure'] *= 1.0 / delayValues[-1]
        qubo += self.penalty_weights['departure'] * subqubos['departure']

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
        qubo += self.penalty_weights['conflict'] * subqubos['conflict']

        print "Calculate departure delay uniqueness contribution"
        subqubos['unique'] = poly()
        for i in range(I):
            Q = poly()
            for a in range(NDelay):
                Q += poly({(var.d[i, a],): 1})
            Q += poly({(): -1})
            subqubos['unique'] += Q * Q
        qubo += self.penalty_weights['unique'] * subqubos['unique']

        if not qubo.isQUBO():
            print "WARNING: Cost function is not quadratic!"
        return qubo, subqubos
