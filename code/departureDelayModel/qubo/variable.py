import numpy as np

import qcfco.arraydict
import qcfco.polynomial
import qcfco.variable


class Variable(qcfco.variable.Variable):

    def __eq__(self, var):
        c = []
        c.append(self.representation == var.representation)
        c.append(self.d == var.d)
        c.append(self.delayValues == var.delayValues)
        c.append(self.flat2Multi == var.flat2Multi)
        c.append(self.NDelay == var.NDelay)
        c.append(self.num_qubits == var.num_qubits)
        return all(c)

    def writeToHDF5(self, group):
        group.create_dataset('representation', data=self.representation)
        # group.create_dataset('d', data=self.d)
        # group.create_dataset('flat2Multi', data=self.flat2Multi)
        group.create_dataset('NDelay', data=self.NDelay)
        group.create_dataset('delayValues', data=self.delayValues)
        group.create_dataset('num_qubits', data=self.num_qubits)
        group.create_dataset('I', data=self.I)

        adict = qcfco.arraydict.ArrayDict()
        for k, val in self.d.items():
            adict[k] = np.array(val)
        subgroupname = 'd'
        subgroup = group.create_group(subgroupname)
        adict.writeToHDF5(subgroup)
        adict = qcfco.arraydict.ArrayDict()
        for k, v in self.flat2Multi.items():
            adict[(k,)] = np.array(v, dtype=[('name', 'S10'), ('integerIndex', 'i4'), ('binaryIndex', 'i4')])
        subgroupname = 'flat2Multi'
        subgroup = group.create_group(subgroupname)
        adict.writeToHDF5(subgroup)

    def readFromHDF5(self, group):
        attributes = ['representation', 'd', 'NDelay', 'delayValues', 'flat2Multi', 'num_qubits']
        if any([i not in group for i in attributes]):
            raise ValueError('Did not find delay dataset in hdf5 file %s' % group.file.filename)
        self.representation = group['representation'].value
        self.NDelay = group['NDelay'].value
        self.delayValues = group['delayValues'].value.tolist()
        self.num_qubits = group['num_qubits'].value
        self.I = group['I'].value
        if 'd' not in group:
            raise ValueError('Did not find group "d" group %s in hdf5 file %s' % (group.name, group.file.filename))
        subgroup = group['d']
        adict = qcfco.arraydict.ArrayDict()
        adict.readFromHDF5(subgroup)
        self.d = {}
        for k, v in adict.dict.items():
            self.d[k] = int(v)
        self.flat2Multi = {}
        if 'flat2Multi' not in group:
            raise ValueError('Did not find group "flat2Multi" group %s in hdf5 file %s' % (group.name, group.file.filename))
        subgroup = group['flat2Multi']
        adict = qcfco.arraydict.ArrayDict()
        adict.readFromHDF5(subgroup)
        for k, v in adict.dict.items():
            self.flat2Multi[k[0]] = (str(v[0]), int(v[1]), int(v[2]))

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
        Q = qcfco.polynomial.Polynomial()
        for a in range(self.NDelay):
            index = self.d[i, a]
            Q.poly[(index,)] = self.delayValues[a]
        return Q

    def decodeBitstring(self, bitstring):
        if len(bitstring) != self.num_qubits:
            raise ValueError('Size mismatch')
        intDelay = np.zeros(self.I, dtype=int)
        for i in range(self.I):
            for a in range(self.NDelay):
                index = self.d[i, a]
                intDelay[i] += bitstring[index] * self.delayValues[a]
        return {'delays': intDelay}
