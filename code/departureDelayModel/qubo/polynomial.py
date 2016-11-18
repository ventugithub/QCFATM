import itertools as it
import re
import yaml
import sys
import numpy as np
import arraydict

class Polynomial:
    """A polynomial of binary variables defined as

        P = P0
            + \sum_{i} P1(i) x_i
            + \sum_{i, j} P2(i, j) x_i x_j
            + \sum_{i, j, k} P3(i, j, k) x_i x_j x_k
            + ...

    where x_i are binary variables.
    In particular, P is stored as a Python dictionary mapping the
    tuple of indices to the value of the coefficient. E.g.

        () --> P0
        (i,) --> P1(i)
        (i, j) --> P2(i, j)
    """
    def __init__(self, d={}):
        self.poly = {}
        for i, v in d.items():
            # force ordered indices
            ii = tuple(sorted(set(i)))
            self.poly[ii] = float(d[i])

    def __add__(self, poly2):
        """ Sum of two polynomials """
        result = self.poly.copy()
        for i, v in poly2.poly.items():
            # force ordered indices
            ii = tuple(sorted(set(i)))
            result[ii] = self.poly.get(i, 0) + v
        return Polynomial(result)

    def __iadd__(self, poly2):
        """ += of two polynomials """
        result = self.poly
        for i, v in poly2.poly.items():
            # force ordered indices
            ii = tuple(sorted(set(i)))
            result[ii] = self.poly.get(i, 0) + v
        return Polynomial(result)

    def __sub__(self, poly2):
        """ Difference of two polynomials """
        result = self.poly.copy()
        for i, v in poly2.poly.items():
            # force ordered indices
            ii = tuple(sorted(set(i)))
            result[ii] = self.poly.get(i, 0) - v
        return Polynomial(result)

    def __isub__(self, poly2):
        """ -= of two polynomials """
        result = self.poly
        for i, v in poly2.poly.items():
            # force ordered indices
            ii = tuple(sorted(set(i)))
            result[ii] = self.poly.get(i, 0) - v
        return Polynomial(result)

    def __eq__(self, poly2):
        """ Compare two polynomials """
        return self.poly == poly2.poly.copy()

    def __rmul__(self, factor):
        """ Product of the polynomial with another one or a scalar from the right """
        return self.__mul__(factor)

    def __mul__(self, factor):
        """ Product of the polynomial with another one or a scalar """
        if isinstance(factor, Polynomial):
            result = {}
            for (i, v), (ii, vv) in it.product(self.poly.items(), factor.poly.items()):
                j = tuple(sorted(set(i + ii)))
                result[j] = result.get(j, 0) + v * vv
            return Polynomial(result)
        else:
            result = {}
            for i, v in self.poly.items():
                result[i] = factor * v
            return Polynomial(result)

    def evaluate(self, x):
        """ Evaluate polynomial """
        r = 0.0
        for k, v in self.poly.items():
            term = v
            for i in range(len(k)):
                term *= x[k[i]]
            r += term
        return r

    def isQUBO(self):
        """Check if polynomial is at most of degree 2"""
        for k in self.poly.keys():
            if len(k) > 2:
                return False
        return True

    def getQUBOConnections(self):
        """Return the edges connection two nodes of the polynomial as a list of 2-tuples"""
        J = {}
        for k, v in self.poly.items():
            if len(k) == 2:
                J[k] = v
        return J

    def getDWaveQUBO(self):
        """Return the QUBO in DWave format, i.e. (i, j) -> Q forall i, j"""
        Q = {}
        offset = 0
        for k, v in self.poly.items():
            if len(k) == 2:
                Q[k] = v
            elif len(k) == 1:
                Q[(k[0], k[0])] = v
            elif len(k) == 0:
                offset = v
            else:
                raise ValueError('Unable to return DWave format for this polynomial. It is not a QUBO')
        return Q, offset

    def getCoefficientRange(self):
        maxLinear = 0
        minLinear = sys.maxint
        maxQuadratic = 0
        minQuadratic = sys.maxint
        for k, v in self.poly.items():
            if len(k) == 2:
                if abs(v) > maxQuadratic:
                    maxQuadratic = abs(v)
                if abs(v) < minQuadratic and v != 0:
                    minQuadratic = abs(v)
            elif len(k) == 1:
                if abs(v) > maxLinear:
                    maxLinear = abs(v)
                if abs(v) < minLinear and v != 0:
                    minLinear = abs(v)
            elif len(k) == 0:
                pass
            else:
                raise ValueError('Unable to linear and quadratic cooefictions for this polynomial. It is not a QUBO')
        return maxLinear/minLinear, maxQuadratic/minQuadratic

    def getNVariables(self):
        """Return the number of variable"""
        l = []
        for k in self.poly.keys():
            for i in k:
                l.append(i)
        v = list(set(l))
        return len(v)

    def fromString(self, s):
        """ Read in polynomial from string of the form:

            9 + 30.1 x1 - 25 x1^3 + 36*x2 + 60 x1 x2 + 36 x2^2 + 18 x1 x3

        WARNING: Experimental status"""

        # remove blanks
        s = s.replace(" ", "").replace("\n", "")
        # remove multiply signs
        s = s.replace("*", "")
        # add delimiter to + and -
        s = s.replace("+", "#+").replace("-", "#-")
        # remove first # if necessary
        s = s.lstrip('#')
        # add + for first term if necessary
        if not s.startswith("-") and not s.startswith('+'):
            s = '+' + s
        # remove all exponents
        s = re.sub('\^[0-9]*', '', s)
        # split polynomial terms
        terms = s.split('#')
        # parse each term
        self.poly = {}
        for t in terms:
            factors = t.split('x')
            if factors[0] in ['+', '-']:
                factors[0] = factors[0] + '1'
            coeff = float(factors[0])
            indices = tuple([int(f) for f in factors[1:]])
            self.poly[indices] = self.poly.get(indices, 0) + coeff

    def toString(self):
        result = ''
        for i, v in self.poly.items():
            result = "%s %+f" % (result,  v)
            for j in i:
                result = "%s x%d" % (result, j)
        return result

    def show(self):
        print "# Binary Polynomial"
        print "# indices value"
        for i, v in sorted(self.poly.items()):
            print "%s %s" % (i, v)

    def save_hdf5(self, filename, groupname='Polynomial', mode='a'):
        adict = arraydict.ArrayDict()
        for k, val in self.poly.items():
            adict[k] = np.array(val)
        adict.save(filename, groupname, mode)

    def load_hdf5(self, filename, groupname='Polynomial'):
        adict = arraydict.ArrayDict()
        adict.load(filename, groupname)
        for k, v in adict.dict.items():
            self.poly[k] = v

    def save_txt(self, filename):
        f = open(filename, 'w')
        yaml.dump(self.poly, f)
        f.close()

    def load_txt(self, filename):
        f = open(filename)
        self.poly = yaml.load(f)
        f.close()
