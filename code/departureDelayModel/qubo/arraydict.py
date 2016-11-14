import h5py
import numpy as np

class ArrayDict():
    """ A dictionary with a tuple of integers as keys and numpy arrays of same dimension as values"""
    def __init__(self, d={}):
        self.dict = {}
        for k, v in d.items():
            self.__setitem__(k, v)

    def __setitem__(self, k, v):
        if not isinstance(k, tuple):
            raise ValueError('Key of ArrayDict must be tuple of integers')
        for i in k:
            if not isinstance(i, int):
                raise ValueError('Key of ArrayDict must be tuple of integers')
        if not isinstance(v, np.ndarray):
            raise ValueError('Value of ArrayDict must be Numpy array')
        if self.dict.values() and self.dict.values()[0].shape != v.shape:
            raise ValueError('Value of ArrayDict must be Numpy array of shape %s not %s' % (str(self.dict.values()[0].shape), str(v.shape)))
        self.dict[k] = v

    def __len__(self):
        return self.dict.__len__()

    def __getitem__(self, k):
        return self.dict[k]

    def __eq__(self, other):
        keq = self.dict.keys() == other.dict.keys()
        veq = np.equal(self.dict.values(), other.dict.values()).all()
        return keq and veq

    def save(self, filename, groupname, mode='w'):
        NItems = len(self)
        # open file
        f = h5py.File(filename, mode)
        # write keys
        dt = h5py.special_dtype(vlen=np.dtype('int32'))
        if groupname not in f:
            group = f.create_group(groupname)
        else:
            group = f[groupname]
        if 'keys' in group:
            del group['keys']
        dt = h5py.special_dtype(vlen=np.dtype('int32'))
        keys = group.create_dataset('keys', (NItems,), dtype=dt)
        n = 0
        for k in self.dict.keys():
            keys[n] = k
            n = n + 1
        # write values
        if 'values' in group:
            del group['values']
        group.create_dataset('values', data=self.dict.values())
        # close file
        f.close()

    def load(self, filename, groupname):
        f = h5py.File(filename, 'r')
        if groupname not in f:
            raise ValueError('Did not find group %s in hdf5 file %s' % (groupname, filename))
        group = f[groupname]
        if 'keys' not in group:
            raise ValueError('Did not find dataset keys in group %s in hdf5 file %s' % (groupname, filename))
        keys = group['keys'].value
        if 'values' not in group:
            raise ValueError('Did not find dataset values in group %s in hdf5 file %s' % (groupname, filename))
        values = group['values'].value
        f.close()

        for key, v in zip(keys, values):
            k = tuple(key.tolist())
            self.dict[k] = v
