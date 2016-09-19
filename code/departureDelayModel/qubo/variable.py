import yaml
import numpy as np
from abc import ABCMeta, abstractmethod

import instance
from polynomial import Polynomial as poly

class IntegerVariable:
    def __init__(self, *args):
        if len(args) == 1:
            self.load(args[0])
        elif len(args) == 2:
            self.update(*args)
        else:
            raise ValueError('Error in instance creation: Wrong number of arguments')

    def update(self, delay, delta):
        self.delay = delay
        self.delta = delta

    def save(self, filename):
        data = {}
        data['delay'] = self.delay.tolist()
        data['delta'] = self.delta.tolist()
        f = open(filename, 'w')
        yaml.dump(data, f)
        f.close()

    def load(self, filename):
        f = open(filename)
        data = yaml.load(f)
        f.close()
        self.delay = np.array(data['delay'], dtype=int)
        self.delta = np.array(data['delta'], dtype=int)

class Variable(object):
    __metaclass__ = ABCMeta

    def __init__(self, *args):
        if len(args) == 1:
            self.update(args[0])
        elif len(args) == 2:
            self.load(args[0], args[1])
        else:
            raise ValueError('Error in instance creation: Wrong number of arguments')

    def __eq__(self, var):
        c = []
        c.append(self.representation == var.representation)
        c.append(self.d == var.d)
        c.append(self.D == var.D)
        c.append(self.delayValues == var.delayValues)
        c.append(self.deltaValues == var.deltaValues)
        c.append(self.flat2Multi == var.flat2Multi)
        c.append(self.NDelay == var.NDelay)
        c.append(self.NDelta == var.NDelta)
        c.append(self.num_qubits == var.num_qubits)
        return all(c)

    def save(self, filename):
        data = {}
        data['representation'] = self.representation
        data['d'] = self.d
        data['D'] = self.D
        data['flat2Multi'] = self.flat2Multi
        data['NDelay'] = self.NDelay
        data['NDelta'] = self.NDelta
        data['delayValues'] = self.delayValues
        data['deltaValues'] = self.deltaValues
        data['num_qubits'] = self.num_qubits
        data['I'] = self.I
        data['K'] = self.K
        f = open(filename, 'w')
        yaml.dump(data, f)
        f.close()

    def load(self, filename, instancefile):
        self.instance = instance.Instance(instancefile)
        f = open(filename)
        data = yaml.load(f)
        f.close()
        self.representation = data['representation']
        self.d = data['d']
        self.D = data['D']
        self.NDelay = data['NDelay']
        self.NDelta = data['NDelta']
        self.delayValues = data['delayValues']
        self.deltaValues = data['deltaValues']
        self.flat2Multi = data['flat2Multi']
        self.num_qubits = data['num_qubits']
        self.I = data['I']
        self.K = data['K']

    @abstractmethod
    def update(self, inst):
        pass

    @abstractmethod
    def delay(self, i):
        pass

    @abstractmethod
    def delta(self, k):
        pass

    @abstractmethod
    def getIntegerVariables(self, bitstring):
        pass

    @abstractmethod
    def getBinaryVariables(self, intDelay, intDelta):
        pass

    @abstractmethod
    def calculateNumberOfVariables(self):
        pass


class Unary(Variable):

    def update(self, inst):
        """ Mapping from multi-indexed integer variables to single-indexed binary variables """
        self.representation = 'unary'
        self.instance = inst
        self.I = len(inst.flights)
        self.K = len(inst.conflicts)
        self.delayValues = list(inst.delays)
        self.deltaValues = np.concatenate((np.sort(-np.array(inst.delays[1:])), np.array(inst.delays))).tolist()
        self.NDelay = len(self.delayValues)
        self.NDelta = len(self.deltaValues)

        ###########################################################################
        # index mapping for delays and arrival time diffences
        ###########################################################################
        # counter for qubits
        num_qubits = 0
        # mapping from d-multi-index (i, a) to flat index
        self.d = {}
        # mapping from Delta-multi-index (k, b) to flat index
        self.D = {}
        # mapping from flat index I to d, D
        self.flat2Multi = {}
        for i in range(self.I):
            for a in range(self.NDelay):
                self.d[i, a] = num_qubits
                self.flat2Multi[num_qubits] = ('d', i, a)
                num_qubits += 1
        for k in range(self.K):
            for a in range(self.NDelta):
                self.D[k, a] = num_qubits
                self.flat2Multi[num_qubits] = ('D', k, a)
                num_qubits += 1
        self.num_qubits = num_qubits

    def delay(self, i):
        Q = poly()
        for a in range(self.NDelay):
            index = self.d[i, a]
            Q.poly[(index,)] = self.delayValues[a]
        return Q

    def delta(self, k):
        Q = poly()
        for a in range(self.NDelta):
            index = self.D[k, a]
            Q.poly[(index,)] = self.deltaValues[a]
        return Q

    def getIntegerVariables(self, bitstring):
        if len(bitstring) != self.num_qubits:
            raise ValueError('Size mismatch')
        intDelay = np.zeros(self.I, dtype=int)
        intDelta = np.zeros(self.K, dtype=int)
        for i in range(self.I):
            for a in range(self.NDelay):
                index = self.d[i, a]
                intDelay[i] += bitstring[index] * self.delayValues[a]
        for k in range(self.K):
            for a in range(self.NDelta):
                index = self.D[k, a]
                intDelta[k] += bitstring[index] * self.deltaValues[a]
        return IntegerVariable(intDelay, intDelta)

    def getBinaryVariables(self, intDelay, intDelta):
        bitstring = np.zeros(self.num_qubits, dtype=int)
        for i in range(self.I):
            a = np.where(self.delayValues == intDelay[i])[0][0]
            index = self.d[i, a]
            bitstring[index] = 1
        for k in range(self.K):
            a = np.where(self.deltaValues == intDelta[k])[0][0]
            index = self.D[k, a]
            bitstring[index] = 1
        return bitstring

    def calculateNumberOfVariables(self):
        r = self.I * self.NDelay
        r = r + self.K * self.NDelta
        return r


class Binary(Variable):

    def update(self, inst):
        """ Mapping from multi-indexed integer variables to single-indexed binary variables """
        self.representation = 'binary'
        self.instance = inst
        self.I = len(inst.flights)
        self.K = len(inst.conflicts)

        # values for delays
        self.delayValues = list(inst.delays)
        # maximum delay value
        self.maxDelay = self.delayValues[-1]
        # check if delay are distributed equidistantly
        if not len(self.delayValues) > 1 or not (np.arange(len(self.delayValues)) == np.array(self.delayValues) / (self.delayValues[1] - self.delayValues[0])).all():
            raise ValueError('Unable to convert to binary variable. Delay Values not equidistant?')
        # delay step size
        self.delayStep = self.delayValues[1] - self.delayValues[0]
        # normalized maximal delay
        self.delayMaxNormalized = self.delayValues[-1] / self.delayStep
        # bit depth for each delay
        self.DelayBitDepth = np.ceil(np.log2(self.delayMaxNormalized + 1)).astype(int)
        # number of binary bits to represent the delay
        self.NBinaryDelay = self.DelayBitDepth - 1
        # number of unary bits to represent the delay
        self.NUnaryDelay = self.delayMaxNormalized - pow(2, self.NBinaryDelay) + 1
        # number of bits for each delay
        self.NDelay = int(self.NBinaryDelay + self.NUnaryDelay)

        # values for deltas
        self.deltaValues = np.concatenate((np.sort(-np.array(inst.delays[1:])), np.array(inst.delays))).tolist()
        # normalized maximal delta
        self.deltaMaxNormalized = 2 * self.delayMaxNormalized
        # bit depth for each delta
        self.DeltaBitDepth = np.ceil(np.log2(self.deltaMaxNormalized + 1)).astype(int)
        # number of binary bits to represent the delta
        self.NBinaryDelta = self.DeltaBitDepth - 1
        # number of unary bits to represent the delta
        self.NUnaryDelta = self.deltaMaxNormalized - pow(2, self.NBinaryDelta) + 1
        # number of bits for each delta
        self.NDelta = int(self.NBinaryDelta + self.NUnaryDelta)

        ###########################################################################
        # index mapping for delays and arrival time diffences
        ###########################################################################
        # counter for qubits
        num_qubits = 0
        # mapping from d-multi-index (i, a) to flat index
        self.d = {}
        # mapping from Delta-multi-index (k, b) to flat index
        self.D = {}
        # mapping from flat index I to d, D
        self.flat2Multi = {}
        for i in range(self.I):
            for a in range(self.NDelay):
                self.d[i, a] = num_qubits
                self.flat2Multi[num_qubits] = ('d', i, a)
                num_qubits += 1
        for k in range(self.K):
            for a in range(self.NDelta):
                self.D[k, a] = num_qubits
                self.flat2Multi[num_qubits] = ('D', k, a)
                num_qubits += 1
        self.num_qubits = num_qubits

    def delay(self, i):
        Q = poly()
        # binary
        for a in range(self.NBinaryDelay):
            index = self.d[i, a]
            Q.poly[(index,)] = 2 ** a * self.delayStep
        # unary
        for a in range(self.NBinaryDelay, self.NDelay):
            index = self.d[i, a]
            Q.poly[(index,)] = a * self.delayStep
        return Q

    def delta(self, k):
        # binary
        Q = poly()
        for a in range(self.NBinaryDelta):
            index = self.D[k, a]
            Q.poly[(index,)] = 2 ** a * self.delayStep
        # unary
        for a in range(self.NBinaryDelta, self.NDelta):
            index = self.D[k, a]
            Q.poly[(index,)] = a * self.delayStep
        Q += poly({(): - self.maxDelay})
        return Q

    def getIntegerVariables(self, bitstring):
        if len(bitstring) != self.num_qubits:
            raise ValueError('Size mismatch')
        intDelay = np.zeros(self.I, dtype=int)
        intDelta = np.zeros(self.K, dtype=int)
        for i in range(self.I):
            # binary
            for a in range(self.NBinaryDelay):
                index = self.d[i, a]
                intDelay[i] += bitstring[index] * pow(2, a)
            # unary
            for a in range(self.NBinaryDelay, self.NDelay):
                index = self.d[i, a]
                intDelay[i] += bitstring[index] * (a - self.NBinaryDelay + 1)
            intDelay[i] *= self.delayStep
        for k in range(self.K):
            # binary
            for a in range(self.NBinaryDelta):
                index = self.D[k, a]
                intDelta[k] += bitstring[index] * pow(2, a)
            # unary
            for a in range(self.NBinaryDelta, self.NDelta):
                index = self.D[k, a]
                intDelta[k] += bitstring[index] * (a - self.NBinaryDelta + 1)
            intDelta[k] *= self.delayStep
            intDelta[k] -= self.maxDelay
        return IntegerVariable(intDelay, intDelta)

    def binary_repr(self, z, nbit):
        return np.array(list(np.binary_repr(z, width=nbit))[::-1], dtype=int)

    def getBinaryVariables(self, intDelay, intDelta):
        bitstring = np.zeros(self.num_qubits, dtype=int)
        for i in range(self.I):
            # binary
            if intDelay[i] / self.delayStep < pow(2, self.NBinaryDelay):
                bits = self.binary_repr(intDelay[i] / self.delayStep, self.NBinaryDelay)
                for a in range(self.NBinaryDelay):
                    index = self.d[i, a]
                    bitstring[index] = bits[a]
                for a in range(self.NBinaryDelay, self.NDelay):
                    index = self.d[i, a]
                    bitstring[index] = 0
            # unary
            else:
                for a in range(self.NBinaryDelay):
                    index = self.d[i, a]
                    bitstring[index] = 1
                a = intDelay[i] / self.delayStep - pow(2, self.NBinaryDelay) + self.NBinaryDelay
                index = self.d[i, a]
                bitstring[index] = 1
        for k in range(self.K):
            # binary
            if (intDelta[k] + self.maxDelay) / self.delayStep < pow(2, self.NBinaryDelta):
                bits = self.binary_repr((intDelta[k] + self.maxDelay) / self.delayStep, self.NBinaryDelta)
                for a in range(self.NBinaryDelta):
                    index = self.D[k, a]
                    bitstring[index] = bits[a]
                for a in range(self.NBinaryDelta, self.NDelta):
                    index = self.D[k, a]
                    bitstring[index] = 0
            # unary
            else:
                for a in range(self.NBinaryDelta):
                    index = self.D[k, a]
                    bitstring[index] = 1
                a = (intDelta[k] + self.maxDelay) / self.delayStep - pow(2, self.NBinaryDelta) + self.NBinaryDelta
                index = self.D[k, a]
                bitstring[index] = 1
        return bitstring

    def calculateNumberOfVariables(self):
        r = self.I * self.NDelay
        r = r + self.K * self.NDelta
        return r
