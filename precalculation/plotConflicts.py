#!/usr/bin/env python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
from mpl_toolkits.basemap import Basemap

def prepareWorldMapPlot():
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
    return map

def addPoints(map, trajectory, markersize=2, color='b', marker='+', linewidth=1, linestyle='-', latitude='latitude', longitude='longitude'):
    """ Plot trajectory points

    Arguments:
        map: basemap object for plotting
        trajectory: Pandas object containing columns for latitude and longitude
        markersize: matplotlib markersize
        marker: matplotlib marker
        linewidth: matplotlib linewidth
        linestyle: matplotlib linestyle
        latitude: name of the latitude column
        longitude: name of the longitude column
    """
    x, y = map(np.array(trajectory[longitude]), np.array(trajectory[latitude]))
    map.plot(x, y, 'b', color=color, linestyle=linestyle, linewidth=linewidth, markersize=markersize, marker=marker)

def addTrajectories(map, trajectories, eastWest=False):
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


def addPointConflicts(map, pointConflicts):
    x, y = map(pointConflicts['lon1'].values, pointConflicts['lat1'].values)
    map.plot(x, y, 'r', markersize=6, marker='<', linestyle='o')
    x, y = map(pointConflicts['lon2'].values, pointConflicts['lat2'].values)
    map.plot(x, y, 'g', markersize=6, marker='>', linestyle='o')

def addParallelConflicts(map, pointConflicts):
    x, y = map(pointConflicts['lon1'].values, pointConflicts['lat1'].values)
    map.plot(x, y, 'r', markersize=6, marker='<', linestyle='o')
    x, y = map(pointConflicts['lon2'].values, pointConflicts['lat2'].values)
    map.plot(x, y, 'g', markersize=6, marker='>', linestyle='o')

def getInvolvedFlights(conflictIndex, pointConflicts, parallelConflicts):
    """ given a conflict index, get both flights involved in the conflict

    Arguments:
        conflictIndex
        pointConflicts
        parallelConflicts

    Returns: flight1, flight2
    """
    NPointConflicts = pointConflicts.index.max()
    NParallelConflicts = parallelConflicts.index.max()
    k = int(conflictIndex)
    NConflicts = NPointConflicts + NParallelConflicts
    if k >= 0 and k < NPointConflicts:
        row = pointConflicts.loc[k]
        flight1 = row['flight1']
        flight2 = row['flight2']
        return flight1, flight2, row
    elif k >= NPointConflicts and k < NConflicts:
        k = k - NPointConflicts
        subset = parallelConflicts.loc[k]
        flight1 = subset['flight1'].iloc[0]
        flight2 = subset['flight2'].iloc[0]
        return flight1, flight2, subset
    else:
        print "Conflict index %i out of range" % k
        print "Point conflict indices range from 0 to", NPointConflicts - 1
        print "Parallel conflict indices range from", NPointConflicts, " to", NParallelConflicts
        exit(0)

def getInvolvedConflicts(flights2Conflicts, flight):
    """ Given a flight index, return the list of conflicts

    Arguments:
        flights2Conflicts: Pandas panel containing the mapping from flight index to conflict indices
        flight: flight index
    returns:
        a list of conflict indices
    """
    return flights2Conflicts.loc[flight].dropna().sort_values('arrivalTime')['conflictIndex'].values.astype(int)

def addConflictPlot(map, conflictIndex, trajectories, pointConflicts, parallelConflicts, red=False):
    """ Given a conflict index, plot the trajectories of the involved flights and the conflicting trajectory points

    Arguments:
        map: basemap object for plotting
        conflictIndex: conflict index
        trajectories: Pandas Dataframe containing all trajectories
        pointConflicts: Pandas Dataframe containing the point conflicts
        parallelConflicts: Pandas Dataframe containing the parallel conflicts
        red: plot all conflict points in red (default false)
    """
    # plot involved flight trajectories
    flight1, flight2, conflictTrajectoryPoints = getInvolvedFlights(conflictIndex, pointConflicts, parallelConflicts)
    addPoints(map, trajectories.loc[flight1], markersize=2, marker='+')
    addPoints(map, trajectories.loc[flight2], markersize=2, marker='+')
    # point conflict
    col = 'r' if red else 'g'
    addPoints(map, conflictTrajectoryPoints, color=col, markersize=6, linewidth=6, marker='>', linestyle='-', latitude='lat1', longitude='lon1')
    addPoints(map, conflictTrajectoryPoints, color='r', markersize=6, linewidth=6, marker='<', linestyle='-', latitude='lat2', longitude='lon2')

def addFlightsAndConflicts(map, flightIndices, trajectories, pointConflicts, parallelConflicts, flights2Conflicts, blue=False, red=False):
    """ Given a flight index, plot the trajectories of the involved flights and the conflicting trajectory points

    Arguments:
        map: basemap object for plotting
        flightIndex: flight index
        trajectories: Pandas Dataframe containing all trajectories
        pointConflicts: Pandas Dataframe containing the point conflicts
        parallelConflicts: Pandas Dataframe containing the parallel conflicts
        flights2Conflicts: Pandas panel containing the mapping from flight index to conflict indices
        blue: plot the trajectory of the flight in blue instead of green (default false)
        red: plot all conflict points in red (default false)
    """
    col = 'b' if blue else 'g'
    # plot trajectory of flight
    for flightIndex in flightIndices:
        conflicts = getInvolvedConflicts(flights2Conflicts, flightIndex)
        for conflictIndex in conflicts:
            addConflictPlot(map, conflictIndex, trajectories, pointConflicts, parallelConflicts, red=red)
        addPoints(map, trajectories.loc[flightIndex], color=col, markersize=6, marker='+')

def addMostInvolvedFlightsAndConflicts(map, nmax, trajectories, pointConflicts, parallelConflicts, flights2Conflicts):
    """ Plot the trajectories and conflicts of flights with the highest number of conflicts

    Arguments:
        map: basemap object for plotting
        nmax: number of flights
        trajectories: Pandas Dataframe containing all trajectories
        pointConflicts: Pandas Dataframe containing the point conflicts
        parallelConflicts: Pandas Dataframe containing the parallel conflicts
        flights2Conflicts: Pandas panel containing the mapping from flight index to conflict indices
    """
    # get the flights with the most number of conflicts
    mostInvolvedFlights = flights2Conflicts.count().T.sort_values('conflictIndex', ascending=False).index[:nmax].values
    addFlightsAndConflicts(map, mostInvolvedFlights, trajectories, pointConflicts, parallelConflicts, flights2Conflicts, blue=True, red=True)

def main():
    parser = argparse.ArgumentParser(description='Calculate point conflicts from trajectory data')
    subparsers = parser.add_subparsers(help="Give a keyword", dest='mode')
    parser.add_argument('--trajectory_file', default='data/TrajDataV2_20120729.txt.csv', help='input file containing the trajectory data with consecutive flight index')

    all_parser = subparsers.add_parser("all", help='Plot all trajectories and raw point conflicts')
    all_parser.add_argument('--raw_point_conflict_file', default='data/TrajDataV2_20120729.txt.rawPointConflicts.csv', help='input file containing the raw point conflicts')
    all_parser.add_argument('--eastwest', action='store_true', help='plot eastbound and westbound flights in different colors')

    conflict_parser = subparsers.add_parser("conflict", help='Plot a special conflicts')
    conflict_parser.add_argument('--info', action='store_true', help='Show info for all conflicts without plotting')
    conflict_parser.add_argument('-k', '--conflictIndex', default=0, help='Conflict index to plot', type=int)
    conflict_parser.add_argument('--point_conflict_file', default='data/TrajDataV2_20120729.txt.pointConflicts.csv', help='input file containing the point conflicts')
    conflict_parser.add_argument('--parallel_conflict_file', default='data/TrajDataV2_20120729.txt.parallelConflicts.csv', help='input file containing the parallel conflicts')

    flight_parser = subparsers.add_parser("flight", help='Plot a special flight including all conflicts')
    group = flight_parser.add_mutually_exclusive_group()
    group.add_argument('-i', '--flightIndex', default=0, help='flight index to plot', type=int)
    group.add_argument('-n', '--numberOfFlightIndices', default=0, help='Plot the n flights which have the most conflicts', type=int)
    flight_parser.add_argument('--point_conflict_file', default='data/TrajDataV2_20120729.txt.pointConflicts.csv', help='input file containing the point conflicts')
    flight_parser.add_argument('--parallel_conflict_file', default='data/TrajDataV2_20120729.txt.parallelConflicts.csv', help='input file containing the parallel conflicts')
    flight_parser.add_argument('--flights2conflicts_file', default='data/TrajDataV2_20120729.txt.flights2Conflicts.h5', help='input file the mapping from flight to conflict indices')

    args = parser.parse_args()

    trajectories = pd.read_csv(args.trajectory_file, index_col='flightIndex')
    if args.mode == 'all':
        rawpointConflicts = pd.read_csv(args.raw_point_conflict_file)
        map = prepareWorldMapPlot()
        addTrajectories(map, trajectories, eastWest=args.eastwest)
        addPointConflicts(map, rawpointConflicts)
        plt.show()

    if args.mode == 'conflict':
        pointConflicts = pd.read_csv(args.point_conflict_file, index_col='conflictIndex')
        parallelConflicts = pd.read_csv(args.parallel_conflict_file, index_col='parallelConflict')
        NPointConflicts = pointConflicts.index.max()
        NParallelConflicts = parallelConflicts.index.max()
        if args.info:
            print "Read point conflicts from ", args.point_conflict_file
            print "Read paraellel conflicts from", args.parallel_conflict_file
            print "Point conflict indices range from 0 to", NPointConflicts - 1
            print "Parallel conflict indices range from", NPointConflicts, " to", NParallelConflicts
        else:
            map = prepareWorldMapPlot()
            addConflictPlot(map, args.conflictIndex, trajectories, pointConflicts, parallelConflicts)
            plt.show()

    if args.mode == 'flight':
        pointConflicts = pd.read_csv(args.point_conflict_file, index_col='conflictIndex')
        parallelConflicts = pd.read_csv(args.parallel_conflict_file, index_col='parallelConflict')
        flights2Conflicts = pd.read_hdf(args.flights2conflicts_file, 'flights2Conflicts')
        map = prepareWorldMapPlot()
        if args.flightIndex:
            addFlightsAndConflicts(map, [args.flightIndex], trajectories, pointConflicts, parallelConflicts, flights2Conflicts)
        else:
            addMostInvolvedFlightsAndConflicts(map, args.numberOfFlightIndices, trajectories, pointConflicts, parallelConflicts, flights2Conflicts)
        plt.show()

if __name__ == "__main__":
    main()
