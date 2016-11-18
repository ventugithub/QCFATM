#!/usr/bin/env python
import sys
sys.path.insert(0, '../code/departureDelayModel/qubo/')
import pandas as pd
import networkx as nx
import numpy as np
import argparse
import os
import h5py

import instance
import analysis

def main():
    parser = argparse.ArgumentParser(description='Create QUBO instances from data', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--maxDelay', default=18, help='maximum delay', type=int)
    parser.add_argument('--delayStep', default=3, help='delay step', type=int)
    parser.add_argument('--input', default='data/TrajDataV2_20120729.txt', help='input file containing the trajectory data with consecutive flight index')
    parser.add_argument('-d', '--mindistance', default=30, help='Minimum distance in nautic miles to qualify as a conflict', type=float)
    parser.add_argument('-t', '--mintime', default=180, help='Minimum time difference in minutes to qualify as a potential conflict', type=int)
    parser.add_argument('--delayPerConflict', default=3, help='Delay introduced by each conflict avoiding maneuver', type=int)
    parser.add_argument('--dthreshold', default=3, help='Minimum time difference in minutes to qualify as a real conflict', type=int)
    parser.add_argument('--maxDepartDelay', default=10, help='Maximum departure delay', type=int)
    parser.add_argument('--pointConflictFile', help='input file containing the point conflicts (overwrites -t and -d)')
    parser.add_argument('--parallelConflictFile', help='input file containing the parallel conflicts (overwrites -t and -d)')
    parser.add_argument('--flights2ConflictsFile', help='input file the mapping from flight to conflict indices (overwrites -t and -d)')
    parser.add_argument('--output', default='instances', help='output folder for instances')

    args = parser.parse_args()

    name = "mindist%05.1f_mintime%03i" % (args.mindistance, args.mintime)
    pointConflictFile = "%s.%s.reducedPointConflicts_delay%03i_thres%03i_depart%03i.csv" % (args.input, name, args.delayPerConflict, args.dthreshold, args.maxDepartDelay)
    parallelConflictFile = "%s.%s.reducedParallelConflicts_delay%03i_thres%03i_depart%03i.csv" % (args.input, name, args.delayPerConflict, args.dthreshold, args.maxDepartDelay)
    if args.maxDelay % args.delayStep != 0:
        print "maximum delay is not a multiple of delay step."
        exit(1)

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    flights2ConflictsFile = "%s.%s.flights2Conflicts_delay%03i_thres%03i_depart%03i.h5" % (args.input, name, args.delayPerConflict, args.dthreshold, args.maxDepartDelay)
    if args.pointConflictFile:
        pointConflictFile = args.pointConflictFile
    if args.parallelConflictFile:
        parallelConflictFile = args.parallelConflictFile
    if args.flights2ConflictsFile:
        flights2ConflictsFile = args.flights2ConflictsFile

    pointConflicts = pd.read_csv(pointConflictFile, index_col='conflictIndex')
    parallelConflicts = pd.read_csv(parallelConflictFile, index_col='parallelConflict')
    flights2Conflicts = pd.read_hdf(flights2ConflictsFile, 'flights2Conflicts')

    G = analysis.getConflictGraph(pointConflicts, parallelConflicts)

    # get partitions
    partitions = nx.connected_components(G)
    p = []
    l = []
    for partition in partitions:
        p.append(list(partition))
        l.append(len(partition))

    conflictsPerPartition = []
    for N, flights in sorted(zip(l, p)):
        conflicts = []
        for f1 in flights:
            conflicts = conflicts + flights2Conflicts[f1].dropna()['conflictIndex'].values.tolist()
        conflicts = list(set(conflicts))
        conflictsPerPartition.append((N, flights, conflicts))

    delays = [int(d) for d in np.arange(0, args.maxDelay + 1, args.delayStep)]
    NPointConflicts = len(pointConflicts)
    count = 0
    for N, flights, conflicts in conflictsPerPartition:
        arrivalTimes = []
        cnfl = []
        for c in [int(n) for n in conflicts]:
            if c < NPointConflicts:
                pc = pointConflicts.loc[c]
                arrivalTimes.append((int(pc.time1), int(pc.time2)))
                cnfl.append((int(pc.flight1), int(pc.flight2)))
            else:
                pc = parallelConflicts.loc[c - NPointConflicts]
                arrivalTimes.append((int(pc.time1.min()), int(pc.time2.min())))
                cnfl.append((int(pc.flight1.iloc[0]), int(pc.flight2.iloc[0])))
        flights = [int(f) for f in flights]
        filename = args.output + "/atm_instance_partition%04i_delayStep%03i_maxDelay%03i.h5" % (count, args.delayStep, args.maxDelay)
        inst = instance.Instance(flights, cnfl, arrivalTimes, delays)
        inst.save_hdf5(filename)
        f = h5py.File(filename, 'a')
        # save metadata
        for arg in vars(args):
            val = getattr(args, arg)
            if val is not None:
                f['atm-instance'].attrs['Precalculation argument: %s' % arg] = val
        f.close()

        count = count + 1

if __name__ == "__main__":
    main()
