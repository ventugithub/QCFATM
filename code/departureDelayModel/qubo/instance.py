import numpy as np
import qcfco.instance


class Instance(qcfco.instance.Instance):
    def update(self, flights, conflicts, timeLimits, delays):
        self.flights = flights
        self.conflicts = conflicts
        self.timeLimits = timeLimits
        self.delays = delays

        self.check()

    def writeToHDF5(self, group):
        """ write instance data to HDF5 group

        group: HDF5 group name where the data is stored
        """
        group.create_dataset('delays', data=np.array(self.delays, dtype=int))
        group.create_dataset('flights', data=np.array(self.flights, dtype=int))
        group.create_dataset('timeLimits', data=np.array(self.timeLimits, dtype=int))
        group.create_dataset('conflicts', data=np.array(self.conflicts, dtype=int))
        group.attrs['Number of flights'] = len(self.flights)
        group.attrs['Number of conflicts'] = len(self.conflicts)

    def readFromHDF5(self, group):
        """ load instance data from HDF5 group

        group: HDF5 group name where the data is stored
        """
        if any([d not in group for d in ['delays', 'flights', 'timeLimits', 'conflicts']]):
            raise ValueError('Did not find all %s datasets in hdf5 file %s' % (group.name, group.file.filename))
        dataset = group['delays']
        self.delays = dataset.value.tolist()
        dataset = group['flights']
        self.flights = dataset.value.tolist()
        dataset = group['timeLimits']
        self.timeLimits = []
        for i, j in dataset.value:
            self.timeLimits.append((i, j))
        dataset = group['conflicts']
        self.conflicts = []
        for k, p in dataset.value:
            self.conflicts.append((k, p))

    def check(self):
        if len(set(self.flights)) != len(self.flights):
            raise ValueError('Error in instance creation: Duplicates in flight numbers')
        flights = []
        for (i, j) in self.conflicts:
            flights.append(i)
            flights.append(j)
        if not set(flights).issubset(set(self.flights)):
            raise ValueError('Error in instance creation: mismatch between conflicts and flight numbers')
        if len(self.conflicts) != len(self.timeLimits):
            raise ValueError('Error in instance creation: dimension mismatch between conflicts and arrival times')
