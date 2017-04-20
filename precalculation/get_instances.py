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


def getNumberOfRealConflictConfigurations(tmin, tmax, maxDelay, dthreshold=3):
    """ given the minimal and maximal time differnence of a conflict point,
    and assuming a delay step size of one minute, calculate the number of
    configurations (d_i - d_j) for which a real conflict will occur """
    dlow = -dthreshold - tmax
    dupp = dthreshold - tmin
    xp = np.linspace(0.0, maxDelay, maxDelay + 1)
    x, y = np.meshgrid(xp, xp)
    z = np.logical_and(x - y >= dlow, x - y <= dupp)
    return np.count_nonzero(z)

def main():
    parser = argparse.ArgumentParser(description='Create Delay Only ATM QUBO instances from data', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--maxDelays', default=[18], nargs='*', help='maximum values of delay variable', type=int)
    parser.add_argument('--delaySteps', default=[3, 6, 9], nargs='*', help='step sizes of the delay variable', type=int)
    parser.add_argument('--delayStepsAllFactors', action='store_true', help='if set,  use all factors of maxDelay as delay step sizes')
    parser.add_argument('--input', default='data/TrajDataV2_20120729.txt', help='input file containing the trajectory data with consecutive flight index')
    parser.add_argument('-d', '--mindistance', default=30, help='Minimum distance in nautic miles to qualify as a conflict', type=float)
    parser.add_argument('-t', '--mintime', default=21, help='Minimum time difference in minutes to qualify as a potential conflict', type=int)
    parser.add_argument('--delayPerConflict', default=0, help='Delay introduced by each conflict avoiding maneuver', type=int)
    parser.add_argument('--dthreshold', default=3, help='Minimum time difference in minutes to qualify as a real conflict', type=int)
    parser.add_argument('--maxDepartDelay', default=18, help='Maximum departure delay in the precalculation', type=int)
    parser.add_argument('--pointConflictFile', help='input file containing the point conflicts (overwrites -t and -d)')
    parser.add_argument('--parallelConflictFile', help='input file containing the parallel conflicts (overwrites -t and -d)')
    parser.add_argument('--flights2ConflictsFile', help='input file the mapping from flight to conflict indices (overwrites -t and -d)')
    parser.add_argument('--output', default='instances', help='output folder for instances')

    subparsers = parser.add_subparsers(help="Give a keyword", dest='mode')

    subparsers.add_parser("connectedComponents", help='Extract instances from the connected components of the conflict graph', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sp_parser = subparsers.add_parser("subpartitions", help='Extract instances from partitioning a single connected component of the conflict graph', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sp_parser.add_argument('-c', '--component', required=True, help='Index of component (ordered by number of flights) which will be divided into subpartitions', type=int)
    sp_parser.add_argument('-k', '--numPart', required=True, help='Number of subpartitions', type=int)

    subparsers.add_parser("getConflictGraph", help='Extract conflict graph only', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()

    name = "mindist%05.1f_mintime%03i" % (args.mindistance, args.mintime)
    pointConflictFile = "%s.%s.reducedPointConflicts_delay%03i_thres%03i_depart%03i.csv" % (args.input, name, args.delayPerConflict, args.dthreshold, args.maxDepartDelay)
    parallelConflictFile = "%s.%s.reducedParallelConflicts_delay%03i_thres%03i_depart%03i.csv" % (args.input, name, args.delayPerConflict, args.dthreshold, args.maxDepartDelay)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    print "Read in precalculation data ..."
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

    print "Extract instances ..."
    if args.mode == 'connectedComponents':
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

        parallelConflicts['timediff'] = parallelConflicts['time1'] - parallelConflicts['time2']
        NPointConflicts = len(pointConflicts)
        count = 0
        for N, flights, conflicts in conflictsPerPartition:
            print "Get instances from connected component", count + 1, " of", len(conflictsPerPartition)
            timeLimits = []
            cnfl = []
            for c in [int(n) for n in conflicts]:
                if c < NPointConflicts:
                    pc = pointConflicts.loc[c]
                    deltaT = int(pc.time1) - int(pc.time2)
                    timeLimits.append((deltaT, deltaT))
                    cnfl.append((int(pc.flight1), int(pc.flight2)))
                else:
                    pc = parallelConflicts.loc[c - NPointConflicts]
                    timeLimits.append((int(pc.timediff.min()), int(pc.timediff.max())))
                    cnfl.append((int(pc.flight1.iloc[0]), int(pc.flight2.iloc[0])))
            flights = [int(f) for f in flights]

            for maxDelay in args.maxDelays:
                if args.delayStepsAllFactors:
                    delaySteps = []
                    for n in range(1, maxDelay + 1):
                        if maxDelay % n == 0:
                            delaySteps.append(maxDelay / n)
                else:
                    delaySteps = args.delaySteps

                for delayStep in delaySteps:
                    if maxDelay % delayStep != 0:
                        print "maximum delay is not a multiple of delay step."
                        exit(1)

                    delays = [int(d) for d in np.arange(0, maxDelay + 1, delayStep)]
                    filename = args.output + "/atm_instance_partition%04i_delayStep%03i_maxDelay%03i.h5" % (count, delayStep, maxDelay)
                    inst = instance.Instance(flights, cnfl, timeLimits, delays)
                    inst.save(filename)
                    f = h5py.File(filename, 'a')
                    # save metadata
                    for arg in vars(args):
                        val = getattr(args, arg)
                        if val is not None:
                            f['Instance'].attrs['Precalculation argument: %s' % arg] = val
                    f.close()

            count = count + 1

    elif args.mode == 'subpartitions':

        import metis
        numPart = args.numPart
        # get conflict graph
        conflictGraph = analysis.getConflictGraph(pointConflicts, parallelConflicts)
        # get connected components
        components = nx.connected_component_subgraphs(conflictGraph)
        # sort connected components by number of flights
        connectedComponents = sorted(list(components), key=lambda x: len(x.nodes()))
        # get component to subdivide
        component = connectedComponents[args.component]

        # Get the number of combinations of delays of both flights which correspond
        # to a real conflict (assuming a delay step of one minute).
        # Then add this number as a edge weight for partitioning
        for i, j in component.edges():
            tmin = component.edge[i][j]['minTimeDiffWithPartner']
            tmax = component.edge[i][j]['minTimeDiffWithPartner']
            nconflict = getNumberOfRealConflictConfigurations(tmin, tmax, args.maxDelay)
            component.edge[i][j]['nconflict'] = nconflict

        # get the partitioning
        edgecut, partition = metis.part_graph(component, nparts=numPart)

        # get subgraphs
        nodes = component.nodes()
        # create networkx graphs
        subgraphs = [nx.Graph()] * len(set(partition))
        # add nodes
        for node, p in zip(nodes, partition):
            subgraphs[p].add_node(node)
        # add edges and attributes
        for i, j in component.edges():
            if partition[nodes.index(i)] == partition[nodes.index(j)]:
                subgraphs[partition[nodes.index(i)]].add_edge(i, j)
                for k, v in component.edge[i][j].items():
                    subgraphs[partition[nodes.index(i)]].edge[i][j][k] = v
        p = []
        l = []
        for subgraph in subgraphs:
            p.append(list(subgraph.nodes()))
            l.append(subgraph.number_of_nodes())

        conflictsPerPartition = []
        for N, flights in sorted(zip(l, p)):
            conflicts = []
            for f1 in flights:
                conflicts = conflicts + flights2Conflicts[f1].dropna()['conflictIndex'].values.tolist()
            conflicts = list(set(conflicts))
            conflictsPerPartition.append((N, flights, conflicts))

        parallelConflicts['timediff'] = parallelConflicts['time1'] - parallelConflicts['time2']
        delays = [int(d) for d in np.arange(0, args.maxDelay + 1, args.delayStep)]
        NPointConflicts = len(pointConflicts)
        count = 0
        for N, flights, conflicts in conflictsPerPartition:
            timeLimits = []
            cnfl = []
            for c in [int(n) for n in conflicts]:
                if c < NPointConflicts:
                    pc = pointConflicts.loc[c]
                    deltaT = int(pc.time1) - int(pc.time2)
                    timeLimits.append((deltaT, deltaT))
                    cnfl.append((int(pc.flight1), int(pc.flight2)))
                else:
                    pc = parallelConflicts.loc[c - NPointConflicts]
                    timeLimits.append((int(pc.timediff.min()), int(pc.timediff.max())))
                    cnfl.append((int(pc.flight1.iloc[0]), int(pc.flight2.iloc[0])))
            flights = [int(f) for f in flights]

            for maxDelay in args.maxDelays:
                if args.delayStepsAllFactors:
                    delaySteps = []
                    for n in range(1, maxDelay + 1):
                        if maxDelay % n == 0:
                            delaySteps.append(maxDelay / n)
                else:
                    delaySteps = args.delaySteps

                for delayStep in delaySteps:
                    if maxDelay % delayStep != 0:
                        print "maximum delay is not a multiple of delay step."
                        exit(1)

                    delays = [int(d) for d in np.arange(0, maxDelay + 1, delayStep)]
                    filename = args.output + "/atm_instance_partition%04i_subpartition%04i_of%04i_edgecut%010i_delayStep%03i_maxDelay%03i.h5" % (args.component, count, numPart, edgecut, delayStep, maxDelay)
                    inst = instance.Instance(flights, cnfl, timeLimits, delays)
                    inst.save(filename)
                    f = h5py.File(filename, 'a')
                    # save metadata
                    for arg in vars(args):
                        val = getattr(args, arg)
                        if val is not None:
                            f['Instance'].attrs['Precalculation argument: %s' % arg] = val
                    f.close()

            count = count + 1

    if args.mode == 'getConflictGraph':
        G = analysis.getConflictGraph(pointConflicts, parallelConflicts)
        if not os.path.exists(args.output):
            os.mkdir(args.output)
        filename = args.output + "/atm_conflict_graph_maxDepartDelayPrecalculation%03i.txt" % (args.mintime - args.dthreshold)
        f = open(filename, 'w')
        for i, j in G.edges():
            f.write("%i %i\n" % (i, j))
        f.close()

if __name__ == "__main__":
    main()
