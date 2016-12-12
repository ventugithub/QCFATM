#!/usr/bin/env python
import unittest
import numpy as np
import conflict

class testRawConflictDetection(unittest.TestCase):
    def test(self):
        """
        Set up test conflicts
                                               ^
                                              /|\
                                               |
                                               9
                                               |
                                               |
                                               8
                                               |
                                               |
                                               7
                                               |
                                flight 3  0----1----2----3
                                               6         |
                                               |         |
                                               |         4
                                               5         |
                                               |         |
                                               |         5
                                               4         |
                                               |         |
                                               |         6
                                               3         |
                                               |         |
                                               |         7----8----9--->
                     flight 1  0----1----2----32---4----5----6----7----8----9--->
                                               |
            latitude                           |
               ^                               1
              /|\                              |
               |                               |
               |                               0
               |
               |                           flight 2
               |
               ------------> longitude

        """
        deltaLon = 0.01
        deltaLat = 0.01
        delta = 0.001
        N = 10
        lons = []
        lats = []
        times = []
        alts = []
        fs = []
        # flight 1 along a fixed latitude
        lons.append(np.linspace(0, (N - 1) * deltaLon, N))
        lats.append(np.linspace(0, 0, N))
        times.append(np.arange(0, N))
        alts.append(np.zeros(N))
        fs.append(1 * np.ones(N))

        # flight 2 along a fixed longitude
        lons.append(np.linspace(3 * deltaLon + delta, 3 * deltaLon + delta, N))
        lats.append(np.linspace(- 2 * deltaLat, (N - 3) * deltaLat, N))
        times.append(np.arange(0, N))
        alts.append(np.zeros(N))
        fs.append(2 * np.ones(N))

        # flight 3
        lons.append(np.linspace(2 * deltaLon + delta, 5 * deltaLon + delta, 4))
        lats.append(np.linspace(4 * deltaLat + delta, 4 * deltaLat + delta, 4))
        lons.append(np.linspace(5 * deltaLon + delta, 5 * deltaLon + delta, 4))
        lats.append(np.linspace(3 * deltaLat + delta, delta, 4))
        lons.append(np.linspace(6 * deltaLon + delta, 7 * deltaLon + delta, 2))
        lats.append(np.linspace(delta, delta, 2))
        times.append(np.arange(0, N))
        alts.append(np.zeros(N))
        fs.append(3 * np.ones(N))

        self.lon = np.concatenate(lons)
        self.lat = np.concatenate(lats)
        self.flight = np.concatenate(fs)
        self.time = np.concatenate(times)
        self.alt = np.concatenate(alts)

        # show trajectories
        # import matplotlib.pyplot as plt
        # plt.plot(self.lon, self.lat, 'o')
        # plt.grid('on')
        # plt.show()

        mindistance = 30
        mintime = 18
        rawPointConflicts = conflict.detectRawConflicts(self.flight, self.time, self.lat, self.lon, self.alt, mindistance, mintime)
        expectedConflicts = [(1, 2), (1, 3), (1, 3), (1, 3), (2, 3)]
        expectedTimes = [(3, 2), (5, 7), (6, 8), (7, 9), (6, 1)]
        self.assertTrue(np.array_equal(np.array(expectedConflicts), np.array(zip(rawPointConflicts.flight1.values, rawPointConflicts.flight2.values))))
        self.assertTrue(np.array_equal(np.array(expectedTimes), np.array(zip(rawPointConflicts.time1.values, rawPointConflicts.time2.values))))
        # reduce coarse grid
        rawPointConflicts2 = conflict.detectRawConflicts(self.flight, self.time, self.lat, self.lon, self.alt, mindistance, mintime, coarseGridLat=np.rad2deg(0.01), coarseGridLon=np.rad2deg(0.01))
        self.assertTrue(rawPointConflicts.equals(rawPointConflicts2))
        # reduce coarse grid so much that no conflict is detected
        rawPointConflicts2 = conflict.detectRawConflicts(self.flight, self.time, self.lat, self.lon, self.alt, mindistance, mintime, coarseGridLat=np.rad2deg(0.0001), coarseGridLon=np.rad2deg(0.0001))
        self.assertFalse(rawPointConflicts.equals(rawPointConflicts2))

        # change time threshold
        mintime = 3
        rawPointConflicts = conflict.detectRawConflicts(self.flight, self.time, self.lat, self.lon, self.alt, mindistance, mintime)
        expectedConflicts = [(1, 2), (1, 3), (1, 3), (1, 3)]
        expectedTimes = [(3, 2), (5, 7), (6, 8), (7, 9)]
        self.assertTrue(np.array_equal(np.array(expectedConflicts), np.array(zip(rawPointConflicts.flight1.values, rawPointConflicts.flight2.values))))
        self.assertTrue(np.array_equal(np.array(expectedTimes), np.array(zip(rawPointConflicts.time1.values, rawPointConflicts.time2.values))))

        # change time threshold
        mintime = 2
        rawPointConflicts = conflict.detectRawConflicts(self.flight, self.time, self.lat, self.lon, self.alt, mindistance, mintime)
        expectedConflicts = [(1, 2)]
        expectedTimes = [(3, 2)]
        self.assertTrue(np.array_equal(np.array(expectedConflicts), np.array(zip(rawPointConflicts.flight1.values, rawPointConflicts.flight2.values))))
        self.assertTrue(np.array_equal(np.array(expectedTimes), np.array(zip(rawPointConflicts.time1.values, rawPointConflicts.time2.values))))

class testConflictParsing(unittest.TestCase):
    def test(self):
        """
        Set up test conflicts
                                               5----6
                                               |    |
                                               |    |
                 flight 2  0----1----2----3----4    7----8----9--->
                     flight 1  0----1----2----3----4----5----6----7----8----9--->

            latitude
               ^
              /|\
               |
               |
               |
               |
               |
               ------------> longitude

        """
        deltaLon = 0.01
        deltaLat = 0.01
        delta = 0.001
        N = 10
        lons = []
        lats = []
        times = []
        alts = []
        fs = []
        # flight 1 along a fixed latitude
        lons.append(np.linspace(0, (N - 1) * deltaLon, N))
        lats.append(np.linspace(0, 0, N))
        times.append(np.arange(0, N))
        alts.append(np.zeros(N))
        fs.append(1 * np.ones(N))

        # flight 2 parallel to flight 2 with detour
        lons.append(np.linspace(-deltaLon + delta, 3 * deltaLon + delta, 5))
        lats.append(np.linspace(delta, delta, 5))
        lons.append(np.linspace(3 * deltaLon + delta, 4 * deltaLon + delta, 2))
        lats.append(np.linspace(deltaLat + delta, deltaLat + delta, 2))
        lons.append(np.linspace(4 * deltaLon + delta, 6 * deltaLon + delta, 3))
        lats.append(np.linspace(delta, delta, 3))
        times.append(np.arange(0, N))
        alts.append(np.zeros(N))
        fs.append(2 * np.ones(N))

        self.lon = np.concatenate(lons)
        self.lat = np.concatenate(lats)
        self.flight = np.concatenate(fs)
        self.time = np.concatenate(times)
        self.alt = np.concatenate(alts)
        # show trajectories
        # import matplotlib.pyplot as plt
        # plt.plot(self.lon, self.lat, 'o')
        # plt.grid('on')
        # plt.show()
        mindistance = 30
        mintime = 18
        rawPointConflicts = conflict.detectRawConflicts(self.flight, self.time, self.lat, self.lon, self.alt, mindistance, mintime)

        # deltaT = 1
        pointConflicts, parallelConflicts = conflict.parseRawPointConflicts(rawPointConflicts, deltaT=1)
        expectedConflictIndices = [0, 0, 0, 0, 1, 1, 1]
        expectedConflicts = [(1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2)]
        expectedTimes = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 7), (5, 8), (6, 9)]
        self.assertTrue(np.array_equal(np.array(expectedConflictIndices), np.array(parallelConflicts.index.values)))
        self.assertTrue(np.array_equal(np.array(expectedConflicts), np.array(zip(parallelConflicts.flight1.values, parallelConflicts.flight2.values))))
        self.assertTrue(np.array_equal(np.array(expectedTimes), np.array(zip(parallelConflicts.time1.values, parallelConflicts.time2.values))))

        # deltaT = 3
        pointConflicts, parallelConflicts = conflict.parseRawPointConflicts(rawPointConflicts, deltaT=3)
        expectedConflictIndices = [0, 0, 0, 0, 0, 0, 0]
        expectedConflicts = [(1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2)]
        expectedTimes = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 7), (5, 8), (6, 9)]
        self.assertTrue(np.array_equal(np.array(expectedConflictIndices), np.array(parallelConflicts.index.values)))
        self.assertTrue(np.array_equal(np.array(expectedConflicts), np.array(zip(parallelConflicts.flight1.values, parallelConflicts.flight2.values))))
        self.assertTrue(np.array_equal(np.array(expectedTimes), np.array(zip(parallelConflicts.time1.values, parallelConflicts.time2.values))))

class testPotentialConflictCalculation(unittest.TestCase):
    def setUp(self):
        """
        Set up test conflicts
                                               ^
                                              /|\
                                               |
                                               9
                                               |
                                               |
                                               8
                                               |
                                               |
                                               7
                                               |
                                flight 3  0----1----2----3
                                               6         |
                                               |         |
                                               |         4
                                               5         |
                                               |         |
                                               |         5
                                               4         |
                                               |         |
                                               |         6
                                               3         |
                                          4----5----6    |
                                          |    |    |    7----8----9--->
                     flight 1  0----1----2|---32---4|---5----6----7----8----9--->
                 flight 4  0----1----2----3    |    7----8----9--->
                                               |
            latitude                           1
               ^                               |
              /|\                              |
               |                               0
               |
               |                           flight 2
               |
               |
               ------------> longitude

        """
        deltaLon = 0.01
        deltaLat = 0.01
        delta = 0.001
        N = 10
        lons = []
        lats = []
        times = []
        alts = []
        fs = []
        # flight 1 along a fixed latitude
        lons.append(np.linspace(0, (N - 1) * deltaLon, N))
        lats.append(np.linspace(0, 0, N))
        times.append(np.arange(0, N))
        alts.append(np.zeros(N))
        fs.append(1 * np.ones(N))

        # flight 2 along a fixed longitude
        lons.append(np.linspace(3 * deltaLon + delta, 3 * deltaLon + delta, N))
        lats.append(np.linspace(- 2 * deltaLat, (N - 3) * deltaLat, N))
        times.append(np.arange(0, N))
        alts.append(np.zeros(N))
        fs.append(2 * np.ones(N))

        # flight 3
        lons.append(np.linspace(2 * deltaLon + delta, 5 * deltaLon + delta, 4))
        lats.append(np.linspace(4 * deltaLat + delta, 4 * deltaLat + delta, 4))
        lons.append(np.linspace(5 * deltaLon + delta, 5 * deltaLon + delta, 4))
        lats.append(np.linspace(3 * deltaLat + delta, delta, 4))
        lons.append(np.linspace(6 * deltaLon + delta, 7 * deltaLon + delta, 2))
        lats.append(np.linspace(delta, delta, 2))
        times.append(np.arange(0, N))
        alts.append(np.zeros(N))
        fs.append(3 * np.ones(N))

        # flight 4
        lons.append(np.linspace(-deltaLon + delta, 2 * deltaLon + delta, 4))
        lats.append(np.linspace(-delta, -delta, 4))
        lons.append(np.linspace(2 * deltaLon + delta, 4 * deltaLon + delta, 3))
        lats.append(np.linspace(deltaLat - delta, deltaLat - delta, 3))
        lons.append(np.linspace(4 * deltaLon + delta, 6 * deltaLon + delta, 3))
        lats.append(np.linspace(-delta, -delta, 3))
        times.append(np.arange(0, N))
        alts.append(np.zeros(N))
        fs.append(4 * np.ones(N))

        self.lon = np.concatenate(lons)
        self.lat = np.concatenate(lats)
        self.flight = np.concatenate(fs)
        self.time = np.concatenate(times)
        self.alt = np.concatenate(alts)

        # show trajectories
        # import matplotlib.pyplot as plt
        # plt.plot(self.lon, self.lat, 'o')
        # plt.grid('on')
        # plt.show()
        mindistance = 30
        mintime = 18
        self.rawPointConflicts = conflict.detectRawConflicts(self.flight, self.time, self.lat, self.lon, self.alt, mindistance, mintime)
        self.pointConflicts, self.parallelConflicts = conflict.parseRawPointConflicts(self.rawPointConflicts, deltaT=1)

    def testRawConflictDetection(self):
        expectedConflicts = [(1, 2), (1, 3), (1, 3), (1, 3), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (2, 3), (2, 4), (3, 4), (3, 4)]
        expectedTimes = [(3, 2), (5, 7), (6, 8), (7, 9), (0, 1), (1, 2), (2, 3), (4, 7), (5, 8), (6, 9), (6, 1), (3, 5), (7, 8), (8, 9)]
        self.assertTrue(np.array_equal(np.array(expectedConflicts), np.array(zip(self.rawPointConflicts.flight1.values, self.rawPointConflicts.flight2.values))))
        self.assertTrue(np.array_equal(np.array(expectedTimes), np.array(zip(self.rawPointConflicts.time1.values, self.rawPointConflicts.time2.values))))

    def testConflictParsing(self):
        expectedConflictIndices = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
        expectedConflicts = [(1, 3), (1, 3), (1, 3), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (3, 4), (3, 4)]
        expectedArrivalTs = [(5, 7), (6, 8), (7, 9), (0, 1), (1, 2), (2, 3), (4, 7), (5, 8), (6, 9), (7, 8), (8, 9)]
        self.assertTrue(np.array_equal(np.array(expectedConflictIndices), np.array(self.parallelConflicts.index.values)))
        self.assertTrue(np.array_equal(np.array(expectedConflicts), np.array(zip(self.parallelConflicts.flight1.values, self.parallelConflicts.flight2.values))))
        self.assertTrue(np.array_equal(np.array(expectedArrivalTs), np.array(zip(self.parallelConflicts.time1.values, self.parallelConflicts.time2.values))))

        expectedConflictIndices = [0, 1, 2]
        expectedConflicts = [(1, 2), (2, 3), (2, 4)]
        expectedArrivalTs = [(3, 2), (6, 1), (3, 5)]
        self.assertTrue(np.array_equal(np.array(expectedConflictIndices), np.array(self.pointConflicts.index.values)))
        self.assertTrue(np.array_equal(np.array(expectedConflicts), np.array(zip(self.pointConflicts.flight1.values, self.pointConflicts.flight2.values))))
        self.assertTrue(np.array_equal(np.array(expectedArrivalTs), np.array(zip(self.pointConflicts.time1.values, self.pointConflicts.time2.values))))

    def testPotentialConflictCalculation(self):
        mindistance = 30
        mintime = 2
        rawPointConflicts = conflict.detectRawConflicts(self.flight, self.time, self.lat, self.lon, self.alt, mindistance, mintime)
        pointConflicts, parallelConflicts = conflict.parseRawPointConflicts(rawPointConflicts, deltaT=1)
        self.assertEqual(pointConflicts.shape[0], 1)
        self.assertEqual(len(parallelConflicts.index.unique()), 2)
        mintime = 3
        rawPointConflicts = conflict.detectRawConflicts(self.flight, self.time, self.lat, self.lon, self.alt, mindistance, mintime)
        pointConflicts, parallelConflicts = conflict.parseRawPointConflicts(rawPointConflicts, deltaT=1)
        self.assertEqual(pointConflicts.shape[0], 2)
        self.assertEqual(len(parallelConflicts.index.unique()), 3)
        mintime = 4
        rawPointConflicts = conflict.detectRawConflicts(self.flight, self.time, self.lat, self.lon, self.alt, mindistance, mintime)
        pointConflicts, parallelConflicts = conflict.parseRawPointConflicts(rawPointConflicts, deltaT=1)
        self.assertEqual(pointConflicts.shape[0], 2)
        self.assertEqual(len(parallelConflicts.index.unique()), 4)
        mintime = 5
        rawPointConflicts = conflict.detectRawConflicts(self.flight, self.time, self.lat, self.lon, self.alt, mindistance, mintime)
        pointConflicts, parallelConflicts = conflict.parseRawPointConflicts(rawPointConflicts, deltaT=1)
        self.assertEqual(pointConflicts.shape[0], 2)
        self.assertEqual(len(parallelConflicts.index.unique()), 4)
        mintime = 6
        rawPointConflicts = conflict.detectRawConflicts(self.flight, self.time, self.lat, self.lon, self.alt, mindistance, mintime)
        pointConflicts, parallelConflicts = conflict.parseRawPointConflicts(rawPointConflicts, deltaT=1)
        self.assertEqual(pointConflicts.shape[0], 3)
        self.assertEqual(len(parallelConflicts.index.unique()), 4)
        mintime = 7
        rawPointConflicts = conflict.detectRawConflicts(self.flight, self.time, self.lat, self.lon, self.alt, mindistance, mintime)
        pointConflicts, parallelConflicts = conflict.parseRawPointConflicts(rawPointConflicts, deltaT=1)
        self.assertEqual(pointConflicts.shape[0], 3)
        self.assertEqual(len(parallelConflicts.index.unique()), 4)

if __name__ == '__main__':
    unittest.main()
