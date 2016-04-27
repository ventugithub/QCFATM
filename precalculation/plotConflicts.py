#!/usr/bin/env python
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from mpl_toolkits.basemap import Basemap

def plotTrajectoriesAndPointConflicts(trajectories, pointConflicts, eastWest=False):
    # Create a figure of size (i.e. pretty big)
    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(1, 1, 1)

    # Create a map, using the Gall-Peters projection,
    map = Basemap(projection='gall',
                  # with low resolution,
                  resolution='l',
                  # And threshold 100000
                  area_thresh=100000.0,
                  # Centered at 0,0 (i.e null island)
                  lat_0=0, lon_0=0)

    # Draw the coastlines on the map
    map.drawcoastlines()

    # Draw country borders on the map
    map.drawcountries()

    # Fill the land with grey
    map.fillcontinents(color='#888888')

    # Draw the map boundaries
    map.drawmapboundary(fill_color='#f4f4f4')

    # Define our longitude and latitude points
    # We have to use .values because of a wierd bug when passing pandas data
    # to basemap.

    if eastWest:
        for flightIndex in set(trajectories.index):
            x, y = map(trajectories[trajectories.index == flightIndex]['longitude'].values, trajectories[trajectories.index == flightIndex]['latitude'].values)
            if x[1] > x[0]:
                map.plot(x, y, 'b', markersize=2, marker='+')
            else:
                map.plot(x, y, color='brown', markersize=2, marker='+')
    else:
        x, y = map(trajectories['longitude'].values, trajectories['latitude'].values)
        map.plot(x, y, 'b', markersize=2, marker='+')

    x, y = map(pointConflicts['lon1'].values, pointConflicts['lat1'].values)
    map.plot(x, y, 'r', markersize=6, marker='<', linestyle='o')
    x, y = map(pointConflicts['lon2'].values, pointConflicts['lat2'].values)
    map.plot(x, y, 'g', markersize=6, marker='>', linestyle='o')
    # Show the map
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Calculate point conflicts from trajectory data')
    parser.add_argument('-t', '--trajectory_file', default='data/TrajDataV2_20120729.txt.csv', help='input file containing the trajectory data with consecutive flight index')
    parser.add_argument('-p', '--point_conflict_file', default='data/TrajDataV2_20120729.txt.pointConflict.csv', help='input file containing the point conflicts')
    parser.add_argument('--eastwest', action='store_true', help='plot eastbound and westbound flights in different colors')
    args = parser.parse_args()

    trajectories = pd.read_csv(args.trajectory_file)
    pointConflicts = pd.read_csv(args.point_conflict_file)
    plotTrajectoriesAndPointConflicts(trajectories, pointConflicts, eastWest=args.eastwest)

if __name__ == "__main__":
    main()
