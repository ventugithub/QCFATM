#!/usr/bin/env python
import pandas as pd
import numpy as np
import conflict
import os
import argparse

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
    pointConflictFile = filename + ".pointConflict.csv"
    if not os.path.exists(pointConflictFile) or not args.use_snapshots:
        pointConflicts = conflict.detectConflicts(trajectories.index, trajectories.time, trajectories.latitude, trajectories.longitude, pointConflictFile, mindistance, mintime)
    else:
        pointConflicts = pd.read_csv(pointConflictFile)
    print pointConflicts.shape[0], "point conflicts detected"

if __name__ == "__main__":
    main()
