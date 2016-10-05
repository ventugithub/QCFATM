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
        self.check()

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
        self.check()

    def check(self):
        if len(set(self.flights)) != len(self.flights):
            raise ValueError('Error in instance creation: Duplicates in flight numbers')
        flights = []
        for (i, j) in self.conflicts:
            flights.append(i)
            flights.append(j)
        if set(flights) != set(self.flights):
            raise ValueError('Error in instance creation: mismatch between conflicts and flight numbers')
        if len(self.conflicts) != len(self.arrivalTimes):
            raise ValueError('Error in instance creation: dimension mismatch between conflicts and arrival times')

