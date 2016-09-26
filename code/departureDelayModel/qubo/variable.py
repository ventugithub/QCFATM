import yaml
import numpy as np
from abc import ABCMeta, abstractmethod

import instance
from polynomial import Polynomial as poly

class IntegerVariable:
    def __init__(self, *args):
        if len(args) == 1:
            if type(args[0]) == str:
                self.load(args[0])
            else:
                self.update(*args)
        else:
            raise ValueError('Error in instance creation: Wrong number of arguments')

    def update(self, delay):
        self.delay = delay

    def save(self, filename):
        data = {}
        data['delay'] = self.delay.tolist()
        f = open(filename, 'w')
        yaml.dump(data, f)
        f.close()

    def load(self, filename):
        f = open(filename)
        data = yaml.load(f)
        f.close()
        self.delay = np.array(data['delay'], dtype=int)

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
        c.append(self.delayValues == var.delayValues)
        c.append(self.flat2Multi == var.flat2Multi)
        c.append(self.NDelay == var.NDelay)
        c.append(self.num_qubits == var.num_qubits)
        return all(c)

    def save(self, filename):
        data = {}
        data['representation'] = self.representation
        data['d'] = self.d
        data['flat2Multi'] = self.flat2Multi
        data['NDelay'] = self.NDelay
        data['delayValues'] = self.delayValues
        data['num_qubits'] = self.num_qubits
        data['I'] = self.I
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
        self.NDelay = data['NDelay']
        self.delayValues = data['delayValues']
        self.flat2Multi = data['flat2Multi']
        self.num_qubits = data['num_qubits']
        self.I = data['I']

    @abstractmethod
    def update(self, inst):
        pass

    @abstractmethod
    def delay(self, i):
        pass

    @abstractmethod
    def getIntegerVariables(self, bitstring):
        pass

    @abstractmethod
    def getBinaryVariables(self, intDelay):
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
        self.delayValues = list(inst.delays)
        self.NDelay = len(self.delayValues)

        ###########################################################################
        # index mapping for delays and arrival time diffences
        ###########################################################################
        # counter for qubits
        num_qubits = 0
        # mapping from d-multi-index (i, a) to flat index
        self.d = {}
        # mapping from flat index I to d
        self.flat2Multi = {}
        for i in range(self.I):
            for a in range(self.NDelay):
                self.d[i, a] = num_qubits
                self.flat2Multi[num_qubits] = ('d', i, a)
                num_qubits += 1
        self.num_qubits = num_qubits

    def delay(self, i):
        Q = poly()
        for a in range(self.NDelay):
            index = self.d[i, a]
            Q.poly[(index,)] = self.delayValues[a]
        return Q

    def getIntegerVariables(self, bitstring):
        if len(bitstring) != self.num_qubits:
            raise ValueError('Size mismatch')
        intDelay = np.zeros(self.I, dtype=int)
        for i in range(self.I):
            for a in range(self.NDelay):
                index = self.d[i, a]
                intDelay[i] += bitstring[index] * self.delayValues[a]
        return IntegerVariable(intDelay)

    def getBinaryVariables(self, intDelay):
        bitstring = np.zeros(self.num_qubits, dtype=int)
        for i in range(self.I):
            a = np.where(self.delayValues == intDelay[i])[0][0]
            index = self.d[i, a]
            bitstring[index] = 1
        return bitstring

    def calculateNumberOfVariables(self):
        r = self.I * self.NDelay
        return r
