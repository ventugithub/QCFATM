import yaml
import h5py
import numpy as np

class Instance:
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
        elif len(args) == 4:
            self.flights = args[0]
            self.conflicts = args[1]
            self.arrivalTimes = args[2]
            self.delays = args[3]
        else:
            raise ValueError('Error in instance creation: Wrong number of arguments')
        self.check()

    def save_txt(self, filename):
        data = {}
        data['flights'] = self.flights
        data['conflicts'] = self.conflicts
        data['arrivalTimes'] = self.arrivalTimes
        data['delays'] = self.delays
        f = open(filename, 'w')
        yaml.dump(data, f)
        f.close()

    def load_txt(self, filename):
        # parse instance file
        f = open(filename)
        data = yaml.load(f)
        f.close()
        self.flights = data['flights']
        self.conflicts = data['conflicts']
        self.arrivalTimes = data['arrivalTimes']
        self.delays = data['delays']
        self.check()

    def save_hdf5(self, filename, name='atm-instance', mode='w'):
        f = h5py.File(filename, mode)
        if name in f:
            del f[name]
        group = f.create_group(name)
        group.create_dataset('delays', data=np.array(self.delays, dtype=int))
        group.create_dataset('flights', data=np.array(self.flights, dtype=int))
        group.create_dataset('arrivalTimes', data=np.array(self.arrivalTimes, dtype=int))
        group.create_dataset('conflicts', data=np.array(self.conflicts, dtype=int))
        group.attrs['Number of flights'] = len(self.flights)
        group.attrs['Number of conflicts'] = len(self.conflicts)
        f.close()

    def load_hdf5(self, filename, name='atm-instance'):
        f = h5py.File(filename, 'r')
        if name not in f:
            raise ValueError('Did not find %s group in hdf5 file %s' % (name, filename))
        group = f[name]
        if any([d not in group for d in ['delays', 'flights', 'arrivalTimes', 'conflicts']]):
            raise ValueError('Did not find all %s datasets in hdf5 file %s' % (name, filename))
        dataset = group['delays']
        self.delays = dataset.value.tolist()
        dataset = group['flights']
        self.flights = dataset.value.tolist()
        dataset = group['arrivalTimes']
        self.arrivalTimes = []
        for i, j in dataset.value:
            self.arrivalTimes.append((i, j))
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
        if len(self.conflicts) != len(self.arrivalTimes):
            raise ValueError('Error in instance creation: dimension mismatch between conflicts and arrival times')
