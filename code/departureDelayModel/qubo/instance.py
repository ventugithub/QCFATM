import yaml

class Instance:
    def __init__(self, *args):
        if len(args) == 1:
            self.load(args[0])
        elif len(args) == 4:
            self.flights = args[0]
            self.conflicts = args[1]
            self.arrivalTimes = args[2]
            self.delays = args[3]
        else:
            raise ValueError('Error in instance creation: Wrong number of arguments')

    def save(self, filename):
        data = {}
        data['flights'] = self.flights
        data['conflicts'] = self.conflicts
        data['arrivalTimes'] = self.arrivalTimes
        data['delays'] = self.delays
        f = open(filename, 'w')
        yaml.dump(data, f)
        f.close()

    def load(self, filename):
        # parse instance file
        f = open(filename)
        data = yaml.load(f)
        f.close()
        self.flights = data['flights']
        self.conflicts = data['conflicts']
        self.arrivalTimes = data['arrivalTimes']
        self.delays = data['delays']

        # check array dimensions
        try:
            assert len(self.conflicts) == len(self.arrivalTimes)
        except AssertionError:
            print ('Dimension mismatch in input file %s' % filename)
            raise
