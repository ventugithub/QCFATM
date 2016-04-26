#!/usr/bin/env python
import pandas as pd
import numpy as np
import conflict
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Calculate point conflicts from trajectory data')
    parser.add_argument('-i', '--input', default='data/TrajDataV2_20120729.txt', help='input file containing the trajectory data')
    parser.add_argument('-o', '--output', default='pointConflicts.dat', help='output file')
    parser.add_argument('-d', '--mindistance', default=30, help='Minimum distance in nautic miles to qualify as a conflict')
    parser.add_argument('-t', '--mintime', default=60, help='Minimum time difference in minutes to qualify as a conflict')
    parser.add_argument('--overwrite', action='store_true', help='Force recalulation of intermediate step of caluclatin consecutive flight indices in the data')
    args = parser.parse_args()

    # nautic mile in kilometers
    nautic = 1.852

    # minimal acceptable distance in kilometers
    mindistance = args.mindistance * nautic

    # minimal acceptable time difference
    mintime = args.mintime

    inputDataFile = args.input
    filename = inputDataFile + ".h5"
    trajectories = None
    if not os.path.exists(filename) or args.overwrite:
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
        trajectories.to_hdf(filename, 'trajectories', mode='w')
    else:
        print "read in trajectories ..."
        trajectories = pd.read_hdf(filename, 'trajectories')

    # calulate point conflicts
    conflict.detectConflicts(trajectories.index, trajectories.time, trajectories.latitude, trajectories.longitude, args.output, mindistance, mintime)

if __name__ == "__main__":
        main()
