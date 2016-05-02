#!/usr/bin/env python
import pandas as pd
import numpy as np
import conflict
import os
import argparse
import pickle

def main():
    parser = argparse.ArgumentParser(description='Calculate point conflicts from trajectory data')
    parser.add_argument('-i', '--input', default='data/TrajDataV2_20120729.txt', help='input file containing the trajectory data')
    parser.add_argument('-o', '--output', help='output file name without suffix')
    parser.add_argument('-d', '--mindistance', default=30, help='Minimum distance in nautic miles to qualify as a conflict')
    parser.add_argument('-t', '--mintime', default=60, help='Minimum time difference in minutes to qualify as a conflict')
    parser.add_argument('--use_snapshots', action='store_true', help='Force recalulation of intermediate step of caluclatin consecutive flight indices in the data')
    args = parser.parse_args()

    # nautic mile in kilometers
    nautic = 1.852

    # minimal acceptable distance in kilometers
    mindistance = args.mindistance * nautic

    # minimal acceptable time difference
    mintime = args.mintime

    inputDataFile = args.input
    filename = inputDataFile
    if args.output:
        filename = args.output
    trajectoryFile = filename + ".csv"
    trajectories = None
    if not os.path.exists(trajectoryFile) or not args.use_snapshots:
        print "Read in trajectories ..."
        trajectories = pd.read_csv(inputDataFile,
                                   delimiter="\t",
                                   names=('flight', 'date', 'wind', 'time', 'speed', 'altitude', 'latitude', 'longitude', 'nan'),
                                   usecols=('flight', 'date', 'wind', 'time', 'speed', 'altitude', 'latitude', 'longitude')
                                   )
        print "Add consecutive flight index to data ..."
        # add consecutive flight index to the data
        flightNames = trajectories['flight'].unique()
        trajectories['flightIndex'] = trajectories['flight'].map(lambda x: np.where(flightNames == x)[0][0])
        # set consecutive flight index as dataset index
        trajectories = trajectories.set_index('flightIndex')
        trajectories.to_csv(trajectoryFile, mode='w')
    else:
        print "read in trajectories ..."
        trajectories = pd.read_csv(trajectoryFile)

    # calulate point conflicts
    rawPointConflictFile = filename + ".rawRawPointConflict.csv"
    if not os.path.exists(rawPointConflictFile) or not args.use_snapshots:
        rawPointConflicts = conflict.detectConflicts(trajectories.index, trajectories.time, trajectories.latitude, trajectories.longitude, mindistance, mintime)
        # save to csv file
        rawPointConflicts.to_csv(rawPointConflictFile, mode='w')
        print "Point conflict data written to", rawPointConflictFile
    else:
        rawPointConflicts = pd.read_csv(rawPointConflictFile, index_col='conflictIndex')
    print rawPointConflicts.shape[0], "raw point conflicts detected"

    # calulate point conflicts
    print "parse conflicts ..."
    pointConflictFile = filename + ".pointConflicts.csv"
    parallelConflictFile = filename + ".parallelConflicts.csv"
    if not os.path.exists(pointConflictFile) or not os.path.exists(parallelConflictFile) or not args.use_snapshots:
        pointConflicts, parallelConflicts = conflict.parsePointConflicts(rawPointConflicts, deltaT=2)
        pointConflicts.to_csv(pointConflictFile, mode='w')
        print "Point conflict data written to", pointConflictFile
        parallelConflicts.to_csv(parallelConflictFile, mode='w')
        print "Parallel conflict data written to", parallelConflictFile
    else:
        pointConflicts = pd.read_csv(pointConflictFile, index_col='conflictIndex')
        parallelConflicts = pd.read_csv(parallelConflictFile, index_col='parallelConflict')
    print pointConflicts.shape[0], "point conflicts identified"
    print len(parallelConflicts.index.unique()), "parallel conflicts involving", parallelConflicts.shape[0], "trajectory points identified"

    # calulate mapping of flight index to temporal sorted conflicts
    flights2ConflictsFile = filename + ".flight2Conflicts.pickle"
    if not os.path.exists(flights2ConflictsFile) or not os.path.exists(parallelConflictFile) or not args.use_snapshots:
        flights2Conflicts = conflict.getFlightConflicts(pointConflicts, parallelConflicts)
        pickle.dump(flights2Conflicts, open(flights2ConflictsFile, 'w'))
    else:
        flights2Conflicts = pickle.load(open(flights2ConflictsFile, 'r'))

if __name__ == "__main__":
    main()
