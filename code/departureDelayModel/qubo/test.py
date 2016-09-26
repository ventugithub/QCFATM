#!/usr/bin/env python
import unittest
import numpy as np
import random
import os

import instance
import polynomial
import solver
import variable

# test instance file IO
class testInstance(unittest.TestCase):
    def setUp(self):
        self.filename = "test_instance.yml"
        np.random.seed(0)
        random.seed(0)

    def testIO(self):
        Nf = 10
        Nc = 5
        flights = [int(f) for f in np.arange(0, Nf)]
        conflicts = []
        for c in range(Nc):
            i = flights[random.randint(1, Nf)]
            j = flights[random.randint(1, Nf)]
            while (j == i):
                j = flights[random.randint(1, Nf)]
            conflicts.append((i, j))
        arrivalTimes = zip([int(t) for t in np.random.randint(1, 20, Nc)], [int(t) for t in np.random.randint(1, 20, Nc)])
        delays = [int(d) for d in np.arange(0, 18 + 1, 3)]

        inst = instance.Instance(flights, conflicts, arrivalTimes, delays)
        inst.save(self.filename)
        inst2 = instance.Instance(self.filename)
        self.assertEqual(flights, inst2.flights)
        self.assertEqual(conflicts, inst2.conflicts)
        self.assertEqual(arrivalTimes, inst2.arrivalTimes)
        self.assertEqual(delays, inst2.delays)

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

# test polynomial class
class testPolynomial(unittest.TestCase):
    def setUp(self):
        # qubo1 : 5 x1 + 6 x2 + 3 x1 x3 + x4 + 3 x2 x4 + 3
        self.Q1 = polynomial.Polynomial({(1,): 5, (2,): 6, (3, 1): 3, (2, 4): 3, (4,): 1, (): 3})
        # qubo2 : 7 x5 + x2 x4 + 3 x3 x4 + x2
        self.Q2 = polynomial.Polynomial({(2,): 1, (2, 4): 1, (3, 4): 3, (5,): 7})
        # qubo3 : 7 x1 + 2 x2 + 3 x3 + x4
        self.Q3 = polynomial.Polynomial({(1,): 7, (2,): 2, (3,): 3, (4,): 1})
        self.filename = "test_qubo.yml"

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def testEqual(self):
        Q = self.Q1
        Q = polynomial.Polynomial({(1,): 7, (2,): 2, (3,): 3, (4,): 1})
        self.assertFalse(Q == self.Q1)

    def testSum(self):
        Q = self.Q1 + self.Q2
        expectedQ = polynomial.Polynomial({(): 3, (1,): 5, (2,): 7, (4,): 1, (5,): 7, (1, 3): 3, (2, 4): 4, (3, 4): 3})
        self.assertEqual(Q, expectedQ)
        self.assertEqual(polynomial.Polynomial({(1,): 5, (2,): 6, (3, 1): 3, (2, 4): 3, (4,): 1, (): 3}), self.Q1)
        Q = self.Q1
        Q += self.Q2
        self.assertEqual(Q, expectedQ)

    def testDifference(self):
        Q = self.Q1 - self.Q2
        expectedQ = polynomial.Polynomial({(): 3, (1,): 5, (2,): 5, (4,): 1, (5,): -7, (1, 3): 3, (2, 4): 2, (3, 4): -3})
        self.assertEqual(Q, expectedQ)
        self.assertEqual(polynomial.Polynomial({(1,): 5, (2,): 6, (3, 1): 3, (2, 4): 3, (4,): 1, (): 3}), self.Q1)
        Q = self.Q1
        Q -= self.Q2
        self.assertEqual(Q, expectedQ)

    def testProduct(self):
        Q = self.Q1 * self.Q2
        expectedQ = polynomial.Polynomial()
        expectedQ.fromString("""3. x2 + 5. x1 x2 + 6. x2^2 + 3. x1 x2 x3 + 4. x2 x4 + 5. x1 x2 x4 +
                       9. x2^2 x4 + 9. x3 x4 + 15. x1 x3 x4 + 18. x2 x3 x4 +
                       3. x1 x2 x3 x4 + 9. x1 x3^2 x4 + 1. x2 x4^2 + 3. x2^2 x4^2 +
                       3. x3 x4^2 + 9. x2 x3 x4^2 + 21. x5 + 35. x1 x5 + 42. x2 x5 +
                       21. x1 x3 x5 + 7. x4 x5 + 21. x2 x4 x5""")
        self.assertEqual(Q, expectedQ)
        Q = self.Q1 * self.Q1
        Q2 = self.Q1
        Q2 *= self.Q1
        self.assertEqual(Q, Q2)

    def testScalarProduct(self):
        Q = 3 * self.Q1
        expectedQ = polynomial.Polynomial({(1,): 15, (2,): 18, (3, 1): 9, (2, 4): 9, (4,): 3, (): 9})
        self.assertEqual(Q, expectedQ)

    def testStringIO(self):
        Q = polynomial.Polynomial()
        Q.fromString("""3. x2 - 5. x1 x12 + 6. x2^2 + 3. x1 x2 x3 + 4. x2 x4 + 5. x1 x2 x4 +
                       9. x2^2 x4 + 9. x3 x4 + 15. x1 x3 x4 + 18. x2 x3 x4 +
                       3. x1 x2 x3 x4 - 9. x1 x3^2 x4 + 1. x2 x4^2 + 3. x2^2 x4^2 +
                       3. x3 x4^2 + 9. x2 x3 x4^2 + 21. x5 + 35. x1 x5 + 42. x2 x5 +
                       21. x1 x3 x5 - 7. x41 x5 + 21. x2 x4 x5 x7""")
        sr = Q.toString()
        Q2 = polynomial.Polynomial()
        Q2.fromString(sr)
        self.assertEqual(Q, Q2)

    def testIsQUBO(self):
        Q = self.Q2 * self.Q3
        self.assertFalse(Q.isQUBO())
        Q = self.Q3 * self.Q3
        self.assertTrue(Q.isQUBO())

    def testIO(self):
        self.Q2.save(self.filename)
        expectedQ = polynomial.Polynomial()
        expectedQ.load(self.filename)
        self.assertEqual(self.Q2, expectedQ)

    def testEvaluate(self):
        # qubo1 : 5 x1 + 6 x2 + 3 x1 x3 + x4 + 3 x2 x4 + 3
        # for (x0, x1, x2, x3, x4) = (0, 1, 0, 1, 1)
        # equal to 5 + 3 + 1 + 3 = 12
        x = [0, 1, 0, 1, 1]
        self.Q1 = polynomial.Polynomial({(1,): 5, (2,): 6, (3, 1): 3, (2, 4): 3, (4,): 1, (): 3})
        self.assertEqual(12, self.Q1.evaluate(x))

    def testNVariables(self):
        self.assertEqual(self.Q1.getNVariables(), 4)
        self.assertEqual(self.Q2.getNVariables(), 4)
        self.assertEqual(self.Q3.getNVariables(), 4)

# test variable class
class testVariable(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        random.seed(0)
        self.filenameUnary = "test_unary.yml"
        # self.filenameBinary = "test_binary.yml"
        self.filenameIntVar = "test_integerVariable.yml"
        self.instancefile = "testdata/test_instance.yml"
        self.inst = instance.Instance(self.instancefile)
        self.varUnary = variable.Unary(self.inst)
        # self.varBinary = variable.Binary(self.inst)
        self.assertEqual(self.varUnary.representation, 'unary')
        # self.assertEqual(self.varBinary.representation, 'binary')

        # fill integer variables randomly
        I = len(self.inst.flights)
        delayValues = [int(d) for d in self.inst.delays]
        NDelay = len(delayValues)
        intDelay = np.zeros(I, dtype=int)
        for i in range(I):
            intDelay[i] = delayValues[random.randint(0, NDelay - 1)]
        self.intVar = variable.IntegerVariable(intDelay)

    def testNumberOfVariables(self):
        N1 = self.varUnary.calculateNumberOfVariables()
        N2 = self.varUnary.num_qubits
        self.assertEqual(N1, N2)
        # N1 = self.varBinary.calculateNumberOfVariables()
        # N2 = self.varBinary.num_qubits
        # self.assertEqual(N1, N2)

    def testIntegerVariable(self):
        self.intVar.save(self.filenameIntVar)
        intVar2 = variable.IntegerVariable(self.filenameIntVar)
        self.assertTrue(np.array_equal(self.intVar.delay, intVar2.delay))

    def testIOUnary(self):
        self.varUnary.save(self.filenameUnary)
        var2 = variable.Unary(self.filenameUnary, self.instancefile)
        self.assertEqual(self.varUnary, var2)

    # def testIOBinary(self):
        # self.varBinary.save(self.filenameBinary)
        # varBinary2 = variable.Binary(self.filenameBinary, self.instancefile)
        # self.assertEqual(self.varBinary, varBinary2)

    def testBackAndForthVariablesUnary(self):
        # project to binary string
        bitstring = self.varUnary.getBinaryVariables(self.intVar.delay)
        # back project
        intVar2 = self.varUnary.getIntegerVariables(bitstring)
        # check
        self.assertTrue(np.array_equal(self.intVar.delay, intVar2.delay))

    # def testBackAndForthVariablesBinary(self):
        # # project to binary string
        # bitstring = self.varBinary.getBinaryVariables(self.intVar.delay)
        # # back project
        # intVar2 = self.varBinary.getIntegerVariables(bitstring)
        # # check
        # self.assertTrue(np.array_equal(self.intVar.delay, intVar2.delay))

    def tearDown(self):
        if os.path.exists(self.filenameUnary):
            os.remove(self.filenameUnary)
        # if os.path.exists(self.filenameBinary):
            # os.remove(self.filenameBinary)
        if os.path.exists(self.filenameIntVar):
            os.remove(self.filenameIntVar)

# test solver class
class testSolver(unittest.TestCase):
    def setUp(self):
        self.qubo = polynomial.Polynomial()
        self.qubo.fromString("""-2 x0 + 3. x2 + 2. x1 - 3 x5 - 5 x3
                             + 5. x1 x2 + 6. x2^2 + 3. x1 x3 + 4. x2 x4 + 5. x2 x4 +
                             9. x2 x4 + 9. x3 x4 - 15. x1 x4 + 18. x3 x4 +
                             21. x1 x5 + 7. x3 x5 - 2. x2 x5 + 7""")
        self.qubo.getDWaveQUBO()
        self.solver = solver.Solver(self.qubo)
        self.solver.calculateEmbedding(3, random_seed=0)
        self.filename = 'testSolverEmbedding.yml'
        self.embeddings = self.solver.embeddings

    def testEmbeddingIO(self):
        self.solver.writeEmbedding(self.filename, eIndex=3)
        self.solver.readEmbedding(self.filename, eIndex=3)
        self.assertEqual(self.embeddings[3], self.solver.embeddings[3])

    def testSolution(self):
        physRawResult, logRawResult, energies, numOccurrences = self.solver.solve(num_reads=1, eIndex=3)
        energy = self.qubo.evaluate(logRawResult[0])
        self.assertEqual(energy, -8.0)
        self.assertTrue((logRawResult[0] == [1, 1, 0, 0, 1, 0]).all())

    def testExactSolution(self):
        r = self.solver.solve_exact()
        energy = self.qubo.evaluate(r['solution'])
        self.assertEqual(r['energy'], energy)
        self.assertEqual(r['energy'], -8.0)
        self.assertEqual(r['solution'], [1, 1, 0, 0, 1, 0])

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

if __name__ == '__main__':
    unittest.main()
