import h5py
import yaml
import numpy as np
from abc import ABCMeta, abstractmethod

import instance
import arraydict
from polynomial import Polynomial as poly

class IntegerVariable:
    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            if type(args[0]) == str:
                if kwargs and kwargs['hdf5'] == False:
                    self.load_txt(args[0])
                else:
                    self.load_hdf5(args[0])
            else:
                self.update(*args)
        elif len(args) == 2:
            if type(args[0]) == str:
                self.load_hdf5(args[0], args[1])
        else:
            raise ValueError('Error in integer variable creation: Wrong number of arguments')

    def update(self, delay):
        self.delay = delay

    def save_txt(self, filename):
        data = {}
        data['delay'] = self.delay.tolist()
        f = open(filename, 'w')
        yaml.dump(data, f)
        f.close()

    def load_txt(self, filename):
        f = open(filename)
        data = yaml.load(f)
        f.close()
        self.delay = np.array(data['delay'], dtype=int)

    def save_hdf5(self, filename, name='IntegerVariable', mode='a'):
        f = h5py.File(filename, mode)
        if name in f:
            del f[name]
        group = f.create_group(name)
        group.create_dataset('delay', data=self.delay)
        f.close()

    def load_hdf5(self, filename, name='IntegerVariable'):
        f = h5py.File(filename, 'r')
        if name not in f:
            raise ValueError('Did not find %s group in hdf5 file %s' % (name, filename))
        group = f[name]
        if 'delay' not in group:
            raise ValueError('Did not find delay dataset in hdf5 file %s' % filename)
        dataset = group['delay']
        self.delay = dataset.value

class Variable(object):
    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            self.update(args[0])
        elif len(args) == 2:
            if kwargs and kwargs['hdf5'] == False:
                self.load_txt(args[0], args[1])
            else:
                self.load_hdf5(args[0], args[1])
        else:
            raise ValueError('Error in variable creation: Wrong number of arguments')

    def __eq__(self, var):
        c = []
        c.append(self.representation == var.representation)
        c.append(self.d == var.d)
        c.append(self.delayValues == var.delayValues)
        c.append(self.flat2Multi == var.flat2Multi)
        c.append(self.NDelay == var.NDelay)
        c.append(self.num_qubits == var.num_qubits)
        return all(c)

    def save_txt(self, filename):
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

    def load_txt(self, filename, instancefile):
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

    def save_hdf5(self, filename, mode='a'):
        f = h5py.File(filename, mode)
        if 'variable' in f:
            del f['variable']
        group = f.create_group('variable')
        group.create_dataset('representation', data=self.representation)
        # group.create_dataset('d', data=self.d)
        # group.create_dataset('flat2Multi', data=self.flat2Multi)
        group.create_dataset('NDelay', data=self.NDelay)
        group.create_dataset('delayValues', data=self.delayValues)
        group.create_dataset('num_qubits', data=self.num_qubits)
        group.create_dataset('I', data=self.I)

        group.create_group('d')
        group.create_group('flat2Multi')
        f.close()

        adict = arraydict.ArrayDict()
        for k, val in self.d.items():
            adict[k] = np.array(val)
        adict.save(filename, 'variable/d', mode='a')
        adict = arraydict.ArrayDict()
        for k, v in self.flat2Multi.items():
            adict[(k,)] = np.array(v, dtype=[('name', 'S10'), ('integerIndex', 'i4'), ('binaryIndex', 'i4')])
        adict.save(filename, 'variable/flat2Multi', mode='a')

    def load_hdf5(self, filename, instancefile):
        self.instance = instance.Instance(instancefile)
        f = h5py.File(filename, 'r')
        if 'variable' not in f:
            raise ValueError('Did not find variable group in hdf5 file %s' % filename)
        group = f['variable']
        attributes = ['representation', 'd', 'NDelay', 'delayValues', 'flat2Multi', 'num_qubits']
        if any([i not in group for i in attributes]):
            raise ValueError('Did not find delay dataset in hdf5 file %s' % filename)
        self.representation = group['representation'].value
        self.NDelay = group['NDelay'].value
        self.delayValues = group['delayValues'].value.tolist()
        self.num_qubits = group['num_qubits'].value
        self.I = group['I'].value
        adict = arraydict.ArrayDict()
        adict.load(filename, 'variable/d')
        self.d = {}
        for k, v in adict.dict.items():
            self.d[k] = int(v)
        self.flat2Multi = {}
        adict = arraydict.ArrayDict()
        adict.load(filename, 'variable/flat2Multi')
        for k, v in adict.dict.items():
            self.flat2Multi[k[0]] = (str(v[0]), int(v[1]), int(v[2]))

    def load(self, filename, instancefile, hdf5=True):
        if hdf5:
            self.load_hdf5(filename, instancefile)
        else:
            self.load_txt(filename, instancefile)

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
