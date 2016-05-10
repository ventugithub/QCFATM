#!/usr/bin/env python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tools
import argparse
from mpl_toolkits.basemap import Basemap
import networkx as nx

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

def arrowplot(axes, x, y, narrs=30, dspace=0.5, direc='pos', hl=0.3, hw=6, c='black'):
    ''' narrs  :  Number of arrows that will be drawn along the curve

        dspace :  Shift the position of the arrows along the curve.
                  Should be between 0. and 1.

        direc  :  can be 'pos' or 'neg' to select direction of the arrows

        hl     :  length of the arrow head

        hw     :  width of the arrow head

        c      :  color of the edge and face of the arrow head
    '''

    # r is the distance spanned between pairs of points
    r = [0]
    for i in range(1, len(x)):
        dx = x[i]-x[i-1]
        dy = y[i]-y[i-1]
        r.append(np.sqrt(dx*dx+dy*dy))
    r = np.array(r)

    # rtot is a cumulative sum of r, it's used to save time
    rtot = []
    for i in range(len(r)):
        rtot.append(r[0:i].sum())
    rtot.append(r.sum())

    # based on narrs set the arrow spacing
    aspace = r.sum() / narrs

    if direc is 'neg':
        dspace = -1.*abs(dspace)
    else:
        dspace = abs(dspace)

    # will hold tuples of x,y,theta for each arrow
    arrowData = []
    # current point on walk along data
    # could set arrowPos to 0 if you want
    # an arrow at the beginning of the curve
    arrowPos = aspace*(dspace)

    ndrawn = 0
    rcount = 1
    while arrowPos < r.sum() and ndrawn < narrs:
        x1, x2 = x[rcount - 1], x[rcount]
        y1, y2 = y[rcount - 1], y[rcount]
        da = arrowPos - rtot[rcount]
        theta = np.arctan2((x2 - x1), (y2 - y1))
        ax = np.sin(theta) * da + x1
        ay = np.cos(theta) * da + y1
        arrowData.append((ax, ay, theta))
        ndrawn += 1
        arrowPos += aspace
        while arrowPos > rtot[rcount+1]:
            rcount += 1
            if arrowPos > rtot[-1]:
                break

    # could be done in above block if you want
    for ax, ay, theta in arrowData:
        # use aspace as a guide for size and length of things
        # scaling factors were chosen by experimenting a bit

        dx0 = np.sin(theta) * hl / 2. + ax
        dy0 = np.cos(theta) * hl / 2. + ay
        dx1 = -1. * np.sin(theta) * hl / 2. + ax
        dy1 = -1. * np.cos(theta) * hl / 2. + ay

        if direc is 'neg':
            ax0 = dx0
            ay0 = dy0
            ax1 = dx1
            ay1 = dy1
        else:
            ax0 = dx1
            ay0 = dy1
            ax1 = dx0
            ay1 = dy0

        axes.annotate('', xy=(ax0, ay0), xycoords='data',
                      xytext=(ax1, ay1), textcoords='data',
                      arrowprops=dict(headwidth=hw, frac=1., ec=c, fc=c))

    axes.plot(x, y, color=c)
    axes.set_xlim(x.min() * .9, x.max() * 1.1)
    axes.set_ylim(y.min() * .9, y.max() * 1.1)

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
    flight1, flight2, conflictTrajectoryPoints = tools.getInvolvedFlights(conflictIndex, pointConflicts, parallelConflicts)
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
        conflicts = tools.getInvolvedConflicts(flights2Conflicts, flightIndex)
        for conflictIndex in conflicts:
            addConflictPlot(map, conflictIndex, trajectories, pointConflicts, parallelConflicts, red=red)
        addPoints(map, trajectories.loc[flightIndex], color=col, markersize=6, marker='+')

def addMostInvolvedFlightsAndConflicts(map, nmin, nmax, trajectories, pointConflicts, parallelConflicts, flights2Conflicts):
    """ Plot the trajectories and conflicts of flights with the highest number of conflicts

    Arguments:
        map: basemap object for plotting
        nmin: minimal index in the list of flights ordered by their conflicts to include
        nmax: maximal index in the list of flights ordered by their conflicts to include
        trajectories: Pandas Dataframe containing all trajectories
        pointConflicts: Pandas Dataframe containing the point conflicts
        parallelConflicts: Pandas Dataframe containing the parallel conflicts
        flights2Conflicts: Pandas panel containing the mapping from flight index to conflict indices
    """
    # get the flights with the most number of conflicts
    mostInvolvedFlights = flights2Conflicts.count().T.sort_values('conflictIndex', ascending=False).index[nmin:nmax + 1].values
    addFlightsAndConflicts(map, mostInvolvedFlights, trajectories, pointConflicts, parallelConflicts, flights2Conflicts, blue=True, red=True)

def getPartitions(graph, partition):
    # number of partitions
    Np = len(set(partition))
    # extract nodes from graph
    nodes = list(set([n1 for n1, n2 in graph] + [n2 for n1, n2 in graph]))
    # create networkx graphs
    G = []
    for n in range(Np):
        G.append(nx.Graph())
    # add nodes
    for node, p in zip(nodes, partition):
        G[p].add_node(node)
    # add edges
    for edge in graph:
        if partition[nodes.index(edge[0])] == partition[nodes.index(edge[1])]:
            G[partition[nodes.index(edge[0])]].add_edge(edge[0], edge[1])
    return G

def plotConflictGraph(pointConflicts, parallelConflicts, nparts=None, partition=None, grid=False, separate=False, connectedComponents=False):
    """ Plot the conflicts as a graph with flights as nodes and conflicts as edges

    Arguments:
        pointConflicts: Pandas Dataframe containing the point conflicts
        parallelConflicts: Pandas Dataframe containing the parallel conflicts
        nparts: number of partitions
        partition: partition number to highlight
        grid: plot each partition as a individual subplot
        separate: plot the whole graph with partitions separated
        connectedComponents: find all connected components and plot them spatially separated
    """
    # get edge tuples defining a graph
    l = pd.concat([parallelConflicts.loc[:, ['flight1', 'flight2']], pointConflicts.loc[:, ['flight1', 'flight2']]]).values.tolist()
    plotGraph(l, nparts=nparts, partition=partition, grid=grid, separate=separate, connectedComponents=connectedComponents)

def plotMultiConflictGraph(multiConflicts, nparts=None, partition=None, grid=False, separate=False, connectedComponents=False, NPointConflicts=False):
    """ Plot the interaction between pairwise conflicts as a graph with pairwise conflicts as nodes

    Arguments:
        multiConflicts: Pandas Dataframe containing the conflicts between pairwise conflicts
        nparts: number of partitions
        partition: partition number to highlight
        grid: plot each partition as a individual subplot
        separate: plot the whole graph with partitions separated
        connectedComponents: find all connected components and plot them spatially separated
        NPointConflicts: number of point conflicts to draw different colors for point and parallel conflicts (default: highlight only greatest cluster)
    """
    # get edge tuples defining a graph
    l = multiConflicts.loc[:, ['conflict1', 'conflict2']].values.tolist()
    if not NPointConflicts:
        plotGraph(l, nparts=nparts, partition=partition, grid=grid, separate=separate, connectedComponents=connectedComponents)
    else:
        nodes = set([n1 for n1, n2 in l] + [n2 for n1, n2 in l])
        node_color = np.array(list(nodes)) >= NPointConflicts
        plotGraph(l, nparts=nparts, partition=partition, grid=grid, separate=separate, connectedComponents=connectedComponents, node_color=node_color)

def plotGraph(edges, nparts=None, partition=None, grid=False, separate=False, connectedComponents=False, node_position=False, node_color='r'):
    """ Plot the a graph

    Arguments:
        edges: list of tuples defining the graph
        nparts: number of partitions
        partition: partition number to highlight
        grid: plot each partition as a individual subplot
        separate: plot the whole graph with partitions separated
        connectedComponents: find all connected components and plot them spatially separated
        node_color: sequence of color values in [0, 1], same length as number of nodes
        node_position: sequence of (x, y) values indicating the position of each node
    """
    l = edges
    # convert to networkx format
    # extract nodes from graph
    nodes = set([n1 for n1, n2 in l] + [n2 for n1, n2 in l])
    # create networkx graph
    G = nx.Graph()
    # add nodes
    for node in nodes:
        G.add_node(node)
    # add edges
    for edge in l:
        G.add_edge(edge[0], edge[1])

    if not nparts and not connectedComponents and not node_position:
        nx.draw(G, pos=nx.spring_layout(G), node_size=100, node_color=node_color)
    elif not connectedComponents and not node_position:
        try:
            import metis
        except:
            print "Unable to plot partitioned graph without metis installed"
            raise

        p = metis.part_graph(G, nparts=nparts)
        partition_color = np.array(p[1])
        if not grid and not separate:
            if partition:
                assert(partition < nparts)
                partition_color = partition_color == partition
            else:
                partition_color = np.array(partition_color)/float(nparts)
            nx.draw(G, node_color=partition_color, node_size=100)
        elif grid:
            fig = plt.figure(figsize=(6, 3*nparts))
            # initial positioning
            init_pos = nx.spring_layout(G)
            ax = []
            for i in range(nparts):
                ax.append(fig.add_subplot(int(0.5 * nparts), 2, i + 1))
                partition_color = np.array(p[1])
                partition_color_i = partition_color == i
                nx.draw(G, node_color=partition_color_i, node_size=100, ax=ax[i], pos=init_pos)
        elif separate:
            graphs = getPartitions(l, p[1])
            partition_color = np.array(p[1])
            # partition_color = np.array(partition_color)/float(nparts)
            perm = np.random.permutation(nparts)
            partition_color = np.apply_along_axis(lambda x: perm[x], axis=0, arr=partition_color)
            layout = {}
            nrow = 3
            ncol = 5
            nmulti = (nparts - nparts % (nrow * ncol)) / (nrow * ncol) + 1
            nrows = nmulti * nrow
            scale = 2
            for n in range(nparts):
                xpos = scale * (n % nrows)
                ypos = scale * (n - n % nrows) / nrows
                d = nx.spring_layout(graphs[n], center=(xpos, ypos))
                layout = dict(layout.items() + d.items())
            nx.draw(G, node_size=100, pos=layout, node_color=partition_color)
    elif connectedComponents:
        compgen = nx.connected_components(G)
        NNodes = len(nodes)
        partition = np.empty((NNodes), dtype=int)
        ipart = 0
        ipart_max = 0
        max_comp = 0
        for c in compgen:
            if len(c) > max_comp:
                max_comp = len(c)
                ipart_max = ipart
            for node in c:
                partition[list(nodes).index(int(node))] = ipart
            ipart = ipart + 1
        nparts = ipart
        partition_color = np.array(partition)
        partition_color = partition_color == ipart_max
        if type(node_color) != str:
            partition_color = node_color
        layout = {}
        nrow = 3
        ncol = 5
        nmulti = (nparts - nparts % (nrow * ncol)) / (nrow * ncol) + 1
        nrows = nmulti * nrow
        scale = 2
        graphs = getPartitions(l, partition)
        for n in range(nparts):
            xpos = scale * (n % nrows)
            ypos = scale * (n - n % nrows) / nrows
            d = nx.spring_layout(graphs[n], center=(xpos, ypos))
            layout = dict(layout.items() + d.items())
        nx.draw(G, node_size=100, pos=layout, node_color=partition_color)
    else:
        nx.draw(G, node_size=100, pos=node_position, node_color=node_color)

def getConflictCluster(pointConflicts, parallelConflicts, nmin=2, nmax=10, plot=True):
    """ Calculate the partition of a given graph with maximal cluster coefficient

    Arguments:
        pointConflicts: Pandas Dataframe containing the point conflicts
        parallelConflicts: Pandas Dataframe containing the parallel conflicts
        nmin: Minimum number of partitions to search for
        nmax: Maximum number of partitions to search for
    """
    # get edge tuples defining a graph
    l = pd.concat([parallelConflicts.loc[:, ['flight1', 'flight2']], pointConflicts.loc[:, ['flight1', 'flight2']]]).values.tolist()
    # convert to networkx format
    # extract nodes from graph
    nodes = set([n1 for n1, n2 in l] + [n2 for n1, n2 in l])
    # create networkx graph
    G = nx.Graph()
    # add nodes
    for node in nodes:
        G.add_node(node)
    # add edges
    for edge in l:
        G.add_edge(edge[0], edge[1])

    try:
        import metis
    except:
        print "Unable search for graph partition without metis installed"
        raise

    maxClusterCoef = 0
    maxClusterGraphs = None
    maxClusterNParts = None
    maxClusterPartitioning = None
    maxClusterPartition = None
    for nparts in range(nmin, nmax + 1):
        p = metis.part_graph(G, nparts=nparts)
        graphs = getPartitions(l, p[1])
        n = 0
        for graph in graphs:
            avclust = nx.average_clustering(graph)
            if avclust > maxClusterCoef:
                maxClusterCoef = avclust
                maxClusterNParts = nparts
                maxClusterGraphs = graphs
                maxClusterPartitioning = p
                maxClusterPartition = n
            n = n + 1

    if plot:
        partition_color = np.array(maxClusterPartitioning[1])
        partition_color = (partition_color == maxClusterPartition)
        layout = {}
        nrow = 3
        ncol = 5
        nmulti = (maxClusterNParts - maxClusterNParts % (nrow * ncol)) / (nrow * ncol) + 1
        nrows = nmulti * nrow
        scale = 2
        for n in range(maxClusterNParts):
            xpos = scale * (n % nrows)
            ypos = scale * (n - n % nrows) / nrows
            d = nx.spring_layout(maxClusterGraphs[n], center=(xpos, ypos))
            layout = dict(layout.items() + d.items())
        nx.draw(G, node_size=100, pos=layout, node_color=partition_color)

    return maxClusterPartitioning[1], maxClusterPartition

def main():
    parser = argparse.ArgumentParser(description='Calculate point conflicts from trajectory data', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(help="Give a keyword", dest='mode')
    parser.add_argument('--trajectory_file', default='data/TrajDataV2_20120729.txt.csv', help='input file containing the trajectory data with consecutive flight index')
    parser.add_argument('--point_conflict_file', default='data/TrajDataV2_20120729.txt.pointConflicts.csv', help='input file containing the point conflicts')
    parser.add_argument('--parallel_conflict_file', default='data/TrajDataV2_20120729.txt.parallelConflicts.csv', help='input file containing the parallel conflicts')
    parser.add_argument('--multi_conflict_file', default='data/TrajDataV2_20120729.txt.multiConflicts.csv', help='input file containing the conflicts between pairwise conflicts')

    all_parser = subparsers.add_parser("all", help='Plot all trajectories and raw point conflicts', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    all_parser.add_argument('--raw_point_conflict_file', default='data/TrajDataV2_20120729.txt.rawPointConflicts.csv', help='input file containing the raw point conflicts')
    all_parser.add_argument('--eastwest', action='store_true', help='plot eastbound and westbound flights in different colors')

    conflict_parser = subparsers.add_parser("conflict", help='Plot a special conflicts', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    conflict_parser.add_argument('--info', action='store_true', help='Show info for all conflicts without plotting')
    conflict_parser.add_argument('-k', '--conflictIndex', default=0, help='Conflict index to plot', type=int)

    flight_parser = subparsers.add_parser("flight", help='Plot a special flight including all conflicts', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    group = flight_parser.add_mutually_exclusive_group()
    group.add_argument('-i', '--flightIndex', default=0, help='flight index to plot', type=int)
    group.add_argument('-n', '--numberOfFlightIndices', default=0, help='Plot the n flights which have the most conflicts', type=int)
    flight_parser.add_argument('--flights2conflicts_file', default='data/TrajDataV2_20120729.txt.flights2Conflicts.h5', help='input file the mapping from flight to conflict indices')

    graph_parser = subparsers.add_parser("graph", help='Plot a conflicting flights as graph', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    graph_parser.add_argument('-n', '--nparts', default=None, help='Number of partitions to plot', type=int)
    graph_parser.add_argument('--multi', action='store_true', help='Plot conflicts between pairwise conflicts instead of pairwise conflicts only')
    group_graph = graph_parser.add_mutually_exclusive_group()
    group_graph.add_argument('-p', '--partition', default=None, help='Partition to highlight', type=int)
    group_graph.add_argument('--grid', action='store_true', help='Plot all partitions in multiple plots')
    group_graph.add_argument('--separate', action='store_true', help='Spatially separate partitions in plot')
    group_graph.add_argument('--component', action='store_true', help='Plot all connected components of the graph')

    subset_parser = subparsers.add_parser("subset", help='Calculate disjunct subset with maximal internal cluster coefficient', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subset_parser.add_argument('-n', '--nmin', default=2, help='Minimal number of partitions to search', type=int)
    subset_parser.add_argument('-m', '--nmax', default=10, help='Maximal number of partitions to search', type=int)
    subset_parser.add_argument('-o', '--output', help='output file name without suffix')

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
    if args.mode == 'graph':
        if not args.multi:
            pointConflicts = pd.read_csv(args.point_conflict_file, index_col='conflictIndex')
            parallelConflicts = pd.read_csv(args.parallel_conflict_file, index_col='parallelConflict')
            plotConflictGraph(pointConflicts, parallelConflicts, nparts=args.nparts, partition=args.partition, separate=args.separate, grid=args.grid, connectedComponents=args.component)
        else:
            pointConflicts = pd.read_csv(args.point_conflict_file, index_col='conflictIndex')
            multiConflicts = pd.read_csv(args.multi_conflict_file, index_col='multiConflictIndex')
            plotMultiConflictGraph(multiConflicts, nparts=args.nparts, partition=args.partition, separate=args.separate, grid=args.grid, connectedComponents=args.component, NPointConflicts=len(pointConflicts))
        plt.show()

    if args.mode == 'subset':
        pointConflicts = pd.read_csv(args.point_conflict_file, index_col='conflictIndex')
        parallelConflicts = pd.read_csv(args.parallel_conflict_file, index_col='parallelConflict')
        partitioning, partition = getConflictCluster(pointConflicts, parallelConflicts, nmin=args.nmin, nmax=args.nmax, plot=True)
        plt.show()

if __name__ == "__main__":
    main()
