#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.lines
import pandas as pd
import numpy as np
import tools
import argparse
from mpl_toolkits.basemap import Basemap
import networkx as nx

def prepareWorldMapPlot(llcrnrlon=None, llcrnrlat=None, urcrnrlon=None, urcrnrlat=None, centerLat=0.0, centerLon=0.0):
    # Create a figure of size (i.e. pretty big)
    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(1, 1, 1)

    # Create a map, using the Gall-Peters projection,
    map = Basemap(projection='gall',
                  # with low resolution,
                  resolution='l',
                  # And threshold 100000
                  area_thresh=100000.0,
                  # Center
                  lat_0=centerLat, lon_0=centerLon,
                  # corners
                  llcrnrlon=llcrnrlon,
                  llcrnrlat=llcrnrlat,
                  urcrnrlon=urcrnrlon,
                  urcrnrlat=urcrnrlat
                  )

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
                      arrowprops=dict(headwidth=hw, ec=c, fc=c))

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
    map.plot(x, y, 'r', markersize=6, marker='<', linestyle='None')
    x, y = map(pointConflicts['lon2'].values, pointConflicts['lat2'].values)
    map.plot(x, y, 'g', markersize=6, marker='>', linestyle='None')

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

def plotConflict(conflictIndex, trajectories, pointConflicts, parallelConflicts, red=False):
    """ Given a conflict index, plot the trajectories of the involved flights and the conflicting trajectory points
    around the conflict region

    Arguments:
        conflictIndex: conflict index
        trajectories: Pandas Dataframe containing all trajectories
        pointConflicts: Pandas Dataframe containing the point conflicts
        parallelConflicts: Pandas Dataframe containing the parallel conflicts
        red: plot all conflict points in red (default false)
    """
    flight1, flight2, conflictTrajectoryPoints = tools.getInvolvedFlights(conflictIndex, pointConflicts, parallelConflicts)
    if isinstance(conflictTrajectoryPoints, pd.core.series.Series):
        minlon = min(conflictTrajectoryPoints.lon1, conflictTrajectoryPoints.lon2) - 1.0
        minlat = min(conflictTrajectoryPoints.lat1, conflictTrajectoryPoints.lat2) - 1.0
        maxlon = max(conflictTrajectoryPoints.lon1, conflictTrajectoryPoints.lon2) + 1.0
        maxlat = max(conflictTrajectoryPoints.lat1, conflictTrajectoryPoints.lat2) + 1.0
        centerLon = 0.5 * (conflictTrajectoryPoints.lon1 + conflictTrajectoryPoints.lon2)
        centerLat = 0.5 * (conflictTrajectoryPoints.lat1 + conflictTrajectoryPoints.lat2)

    elif isinstance(conflictTrajectoryPoints, pd.core.frame.DataFrame):
        minlon = min(conflictTrajectoryPoints.lon1.min(), conflictTrajectoryPoints.lon2.min()) - 1.0
        minlat = min(conflictTrajectoryPoints.lat1.min(), conflictTrajectoryPoints.lat2.min()) - 1.0
        maxlon = max(conflictTrajectoryPoints.lon1.max(), conflictTrajectoryPoints.lon2.max()) + 1.0
        maxlat = max(conflictTrajectoryPoints.lat1.max(), conflictTrajectoryPoints.lat2.max()) + 1.0
        centerLon = 0.5 * (conflictTrajectoryPoints.lon1.mean() + conflictTrajectoryPoints.lon2.mean())
        centerLat = 0.5 * (conflictTrajectoryPoints.lat1.mean() + conflictTrajectoryPoints.lat2.mean())

    else:
        raise ValueError('plotConflict: conflictTrajectoryPoints is neither pandas series nor dataframe')

    # Create a figure of size (i.e. pretty big)
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)

    # Create a map, using the Gall-Peters projection,
    map = Basemap(ax=ax, projection='gall',
                  # with low resolution,
                  resolution='l',
                  # And threshold 100000
                  area_thresh=100000.0,
                  # Center
                  lat_0=centerLat, lon_0=centerLon,
                  # corners
                  llcrnrlon=minlon,
                  llcrnrlat=minlat,
                  urcrnrlon=maxlon,
                  urcrnrlat=maxlat
                  )

    # Draw the coastlines on the map
    map.drawcoastlines()

    # Draw country borders on the map
    map.drawcountries()

    # Fill the land with grey
    map.fillcontinents(color='#888888')

    # Draw the map boundaries
    map.drawmapboundary(fill_color='#f4f4f4')
    # plot involved flight trajectories
    traj1 = trajectories.loc[flight1]
    traj2 = trajectories.loc[flight2]
    traj1 = traj1[traj1.longitude >= minlon]
    traj1 = traj1[traj1.longitude <= maxlon]
    traj1 = traj1[traj1.latitude >= minlat]
    traj1 = traj1[traj1.latitude <= maxlat]
    traj2 = traj2[traj2.longitude >= minlon]
    traj2 = traj2[traj2.longitude <= maxlon]
    traj2 = traj2[traj2.latitude >= minlat]
    traj2 = traj2[traj2.latitude <= maxlat]
    x, y = map(np.array(traj1['longitude']), np.array(traj1['latitude']))
    arrowplot(ax, x, y)
    x, y = map(np.array(traj2['longitude']), np.array(traj2['latitude']))
    arrowplot(ax, x, y)
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

def addMostInvolvedFlightsAndConflicts(map, nfmin, nfmax, trajectories, pointConflicts, parallelConflicts, flights2Conflicts):
    """ Plot the trajectories and conflicts of flights with the highest number of conflicts

    Arguments:
        map: basemap object for plotting
        nfmin: minimal index in the list of flights ordered by their conflicts to include
        nfmax: maximal index in the list of flights ordered by their conflicts to include
        trajectories: Pandas Dataframe containing all trajectories
        pointConflicts: Pandas Dataframe containing the point conflicts
        parallelConflicts: Pandas Dataframe containing the parallel conflicts
        flights2Conflicts: Pandas panel containing the mapping from flight index to conflict indices
    """
    # get the flights with the most number of conflicts
    mostInvolvedFlights = flights2Conflicts.count().T.sort_values('conflictIndex', ascending=False).index[nfmin:nfmax + 1].values
    addFlightsAndConflicts(map, mostInvolvedFlights, trajectories, pointConflicts, parallelConflicts, flights2Conflicts, blue=True, red=True)

def getPartitions(G, partition):
    # number of partitions
    Np = len(set(partition))
    graphs = []
    for n in range(Np):
        graphs.append(nx.Graph())
    # get list of nodes
    lnodes = list(G.nodes())
    # add nodes
    for i in range(len(lnodes)):
        graphs[partition[i]].add_node(lnodes[i])
    # add edges
    for edge in G.edges():
        if partition[lnodes.index(edge[0])] == partition[lnodes.index(edge[1])]:
            graphs[partition[lnodes.index(edge[0])]].add_edge(edge[0], edge[1])
    return graphs

def getConflictGraph(pointConflicts, parallelConflicts):
    """ Get the conflicts as a graph with flights as nodes and conflicts as edges

    Arguments:
        pointConflicts: Pandas Dataframe containing the point conflicts
        parallelConflicts: Pandas Dataframe containing the parallel conflicts
    Returns:
        networkx graph
    """

    # reduce parallel conflicts to one row for each flight combination
    # keep the one with the lowest absolute time difference
    plc = parallelConflicts.loc[:, ['flight1', 'flight2', 'time1', 'time2']].reset_index(level='parallelConflict')
    plc['deltaT'] = np.abs(plc.time1-plc.time2)
    plc = plc.sort_values(by=['parallelConflict', 'flight1', 'flight2', 'deltaT'])
    plc = plc.drop_duplicates(['parallelConflict', 'flight1', 'flight2'])
    plc = plc.set_index('parallelConflict', drop=True)
    plc = plc.sort_values(by=['flight1', 'flight2', 'deltaT'])
    plc = plc.drop_duplicates(['flight1', 'flight2'])
    plc.loc[:, 'isParallelConflict'] = True

    # reduce point conflicts to one row for each flight combination
    # and add absolute time difference to the data
    poc = pointConflicts.loc[:, ['flight1', 'flight2', 'time1', 'time2']]
    poc['deltaT'] = np.abs(poc.time1-poc.time2)
    poc.sort_values(by=['flight1', 'flight2'])
    poc = poc.drop_duplicates(['flight1', 'flight2'])
    poc.loc[:, 'isParallelConflict'] = False

    # concatenate point and parallel conflicts
    conflicts = pd.concat([poc.loc[:, ['flight1', 'flight2', 'deltaT', 'isParallelConflict']], plc.loc[:, ['flight1', 'flight2', 'deltaT', 'isParallelConflict']]])
    # reduce conflicts to one row for each flight combination and set the
    # conflict type to 'parallel', 'point' or 'mixed'
    grouped = conflicts.groupby([conflicts['flight1'], conflicts['flight2']])

    def getType(arr):
        if all(arr):
            return 1.0
        elif not any(arr):
            return 0.0
        else:
            return 0.5
    conflicts = grouped.agg({'deltaT': min, 'isParallelConflict': getType})
    conflicts.reset_index(level=['flight1', 'flight2'], inplace=True)

    # get edge tuples defining a graph
    l = conflicts.loc[:, ['flight1', 'flight2']].values.tolist()
    # convert to networkx format
    # extract nodes from graph
    nodes = np.unique(np.array([n1 for n1, n2 in l] + [n2 for n1, n2 in l]))
    # edge weights
    weights = conflicts['deltaT'].values.tolist()
    # edge colors
    conflictType = conflicts['isParallelConflict'].values.tolist()
    # graph info
    info = {}
    info['nodeColorName'] = 'flights'
    info['edgeColorName'] = 'conflict type'
    info['edgeColorValues'] = {0.0: 'point', 0.5: 'mixed', 1.0: 'parallel'}
    info['edgeWeightName'] = 'time difference'
    info['edgeWeightValues'] = list(set(weights))
    # create networkx graph
    G = nx.Graph(**info)
    # add nodes
    for node in nodes:
        G.add_node(node)
    # add edges
    for i in range(len(l)):
        edge = l[i]
        G.add_edge(edge[0], edge[1], weight=weights[i], color=conflictType[i])
    return G

def getMultiConflictGraph(multiConflicts):
    """ Get the interaction between pairwise conflicts as a graph with pairwise conflicts as nodes

    Arguments:
        multiConflicts: Pandas Dataframe containing the conflicts between pairwise conflicts
    """
    # get edge tuples defining a graph
    def getConflictType(arr):
        if all(arr == 1):
            return 1.0
        elif all(arr == 0):
            return 0.0
        else:
            return 0.5
    grouped = multiConflicts.groupby(['conflict1', 'conflict2'])
    conflicts = grouped.agg({'deltaTMin': min, 'multiConflictType': getConflictType, 'conflictType1': getConflictType, 'conflictType2': getConflictType})
    conflicts.reset_index(level=['conflict1', 'conflict2'], inplace=True)

    # get edge tuples defining a graph
    l = conflicts.loc[:, ['conflict1', 'conflict2']].values.tolist()
    # convert to networkx format
    # extract nodes from graph
    nodes = np.array([n1 for n1, n2 in l] + [n2 for n1, n2 in l])
    # get conflict type of each pair-conflict
    pairConflictTypes = conflicts.loc[:, ['conflictType1', 'conflictType2']].values.tolist()
    nodecolor = np.array([n1 for n1, n2 in pairConflictTypes] + [n2 for n1, n2 in pairConflictTypes])
    # edge weights
    weights = conflicts['deltaTMin'].values.tolist()
    # edge colors
    conflictType = conflicts['multiConflictType'].values.tolist()
    # graph info
    info = {}
    info['nodeColorName'] = 'conflict type'
    info['nodeColorValues'] = {0.0: 'point', 0.5: 'mixed', 1.0: 'parallel'}
    info['edgeColorName'] = 'multi conflict type'
    info['edgeColorValues'] = {0.0: 'point', 0.5: 'mixed', 1.0: 'parallel'}
    info['edgeWeightName'] = 'time difference'
    info['edgeWeightValues'] = list(set(weights))
    # create networkx graph
    G = nx.Graph(**info)
    # add nodes
    for i in range(len(nodes)):
        G.add_node(nodes[i], color=nodecolor[i])
    # add edges
    for i in range(len(l)):
        edge = l[i]
        G.add_edge(edge[0], edge[1], weight=weights[i], color=conflictType[i])
    return G

def plotGraph(G, nparts=None, partition=None, grid=False, separate=False, connectedComponents=False, font_size=8):
    """ Plot the a graph

    Arguments:
        edges: list of tuples defining the graph
        nparts: number of partitions
        partition: partition number to highlight
        grid: plot each partition as highlighted region in the whole graph as a individual subplot
        separate: plot the whole graph with partitions separated
        connectedComponents: find all connected components and plot them spatially separated
        font_size: font size (default: 8)
    """

    # extract weights and color for edges and nodes
    # default values
    width = 1.0
    color = 'r'
    node_color = 'r'
    width_max = 3.0
    width_min = 0.5

    if G.edges():
        if 'weight' in G.edges(data=True)[0][2]:
            weights = [l[2] for l in list(G.edges_iter(data='weight'))]
            # normalize
            width = np.array(weights)
            width = (width_max - width_min) * ((np.max(weights) - weights) / (np.max(weights) - np.min(weights))) + width_min
        if 'color' in G.edges(data=True)[0][2]:
            color = [l[2] for l in list(G.edges_iter(data='color'))]
    hasNodeColor = False
    if G.nodes() and 'color' in G.nodes(data=True)[0][1]:
        hasNodeColor = True
        node_color = [l[1]['color'] for l in list(G.nodes_iter(data='color'))]

    partition_colors = []
    if not nparts and not connectedComponents:
        nx.draw_networkx(G, pos=nx.spring_layout(G), node_size=300, node_color=node_color, font_size=font_size, width=width, edge_color=color)
    elif nparts:
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
            current_node_color = node_color if hasNodeColor else partition_color
            partition_colors = partition_color
            nx.draw_networkx(G, node_color=current_node_color, node_size=300, font_size=font_size, width=width, edge_color=color)
        elif grid:
            fig = plt.figure(figsize=(6, 3*nparts))
            # initial positioning
            init_pos = nx.spring_layout(G)
            ax = []
            npartsEven = (nparts % 2 == 1) + nparts
            for i in range(nparts):
                ax.append(fig.add_subplot(int(0.5 * npartsEven), 2, i + 1))
                partition_color = np.array(p[1])
                partition_color_i = partition_color == i
                current_node_color = node_color if hasNodeColor else partition_color_i
                partition_colors.append(partition_color_i)
                nx.draw_networkx(G, node_color=current_node_color, node_size=300, ax=ax[i], pos=init_pos, font_size=font_size, width=width, edge_color=color)
        elif separate:
            graphs = getPartitions(G, p[1])
            partition_color = np.array(p[1])
            # partition_color = np.array(partition_color)/float(nparts)
            perm = np.random.permutation(nparts)
            partition_color = np.apply_along_axis(lambda x: perm[x], axis=0, arr=partition_color)
            colormax = np.max(np.array(partition_color))
            partition_color = np.array(partition_color) / float(colormax)
            layout = {}
            nrow = 3
            ncol = 5
            nmulti = (nparts - nparts % (nrow * ncol)) / (nrow * ncol) + 1
            nrows = nmulti * nrow
            scale = 4
            for n in range(nparts):
                xpos = scale * (n % nrows)
                ypos = scale * (n - n % nrows) / nrows
                d = nx.circular_layout(graphs[n], center=(xpos, ypos), scale=1.0)
                layout = dict(layout.items() + d.items())
            current_node_color = node_color if hasNodeColor else partition_color
            partition_colors = partition_color
            nx.draw_networkx(G, node_size=300, pos=layout, node_color=current_node_color, font_size=font_size, width=width, edge_color=color)
    elif connectedComponents:
        compgen = nx.connected_components(G)
        NNodes = len(G.nodes())
        partitionList = np.empty((NNodes), dtype=int)
        ipart = 0
        ipart_max = 0
        max_comp = 0
        for c in compgen:
            if len(c) > max_comp:
                max_comp = len(c)
                ipart_max = ipart
            for node in c:
                partitionList[list(G.nodes()).index(int(node))] = ipart
            ipart = ipart + 1
        nparts = ipart
        partition_color = np.array(partitionList)
        partition_color = partition_color == ipart_max
        layout = {}
        nrow = 3
        ncol = 5
        nmulti = (nparts - nparts % (nrow * ncol)) / (nrow * ncol) + 1
        nrows = nmulti * nrow
        scale = 2
        graphs = getPartitions(G, partitionList)
        for n in range(nparts):
            xpos = scale * (n % nrows)
            ypos = scale * (n - n % nrows) / nrows
            d = nx.spring_layout(graphs[n], center=(xpos, ypos))
            layout = dict(layout.items() + d.items())
        current_node_color = node_color if hasNodeColor else partition_color
        partition_colors = partition_color
        nx.draw_networkx(G, node_size=300, pos=layout, node_color=current_node_color, font_size=font_size, width=width, edge_color=color)
    else:
        nx.draw_networkx(G, pos=nx.spring_layout(G), node_size=300, node_color=node_color, font_size=font_size, width=width, edge_color=color)

    ###################################################################################
    # Add legend
    ###################################################################################
    # container for legend entries
    labels = []

    # get graph information
    info = G.graph
    hasNodeColorInfo = 'nodeColorName' in info and 'nodeColorValues' in info
    hasEdgeColorInfo = 'edgeColorName' in info and 'edgeColorValues' in info
    hasEdgeWeightInfo = 'edgeWeightName' in info and 'edgeWeightValues' in info
    if hasNodeColorInfo:
        nodeColorName = info['nodeColorName']
        nodeColorValues = info['nodeColorValues']
        # create legend entry
        sm = matplotlib.cm.ScalarMappable()
        rgb_nodecolor = sm.to_rgba(np.unique(np.array(node_color)))
        label_nodecolor = ["%s = %s" % (nodeColorName, nodeColorValues[c]) for c in set(node_color)]
        for i in range(len(rgb_nodecolor)):
            labels.append(matplotlib.lines.Line2D([], [], color=rgb_nodecolor[i], marker='o', markersize=15, linewidth=0, label=label_nodecolor[i]))
    elif len(partition_colors):
        partition_colors = list(np.unique(np.array(partition_colors)))
        sm = matplotlib.cm.ScalarMappable()
        rgb_color = sm.to_rgba(partition_colors)
        if grid or partition or connectedComponents:
            if grid:
                labelname1 = 'other partitions'
                labelname2 = 'highlighted partition'
            elif partition:
                labelname1 = 'other partitions'
                labelname2 = 'partition %i' % partition
            elif connectedComponents:
                labelname1 = 'other partitions'
                labelname2 = 'most connected partition'
            if 'nodeColorName' in info:
                labelname1 = "%s,  %s" % (info['nodeColorName'], labelname1)
                labelname2 = "%s,  %s" % (info['nodeColorName'], labelname2)
            labels.append(matplotlib.lines.Line2D([], [], color=rgb_color[0], marker='o', markersize=15, linewidth=0, label=labelname1))
            labels.append(matplotlib.lines.Line2D([], [], color=rgb_color[1], marker='o', markersize=15, linewidth=0, label=labelname2))
        else:
            for i in range(len(partition_colors)):
                if 'nodeColorName' in info:
                    labelname = "%s, partition %i" % (info['nodeColorName'], i)
                else:
                    labelname = "partition %i" % i
                labels.append(matplotlib.lines.Line2D([], [], color=rgb_color[i], marker='o', markersize=15, linewidth=0, label=labelname))
    elif 'nodeColorName' in info:
        nodeColorName = info['nodeColorName']
        labels.append(matplotlib.lines.Line2D([], [], color=node_color, marker='o', markersize=15, linewidth=0, label=nodeColorName))
    if hasEdgeColorInfo:
        edgeColorName = info['edgeColorName']
        edgeColorValues = info['edgeColorValues']
        # create legend entry
        colorValues = np.sort(np.unique(np.array(color)))
        sm = matplotlib.cm.ScalarMappable()
        rgb_color = sm.to_rgba(colorValues)
        label_color = ["%s = %s" % (edgeColorName, edgeColorValues[c]) for c in colorValues]
        for i in range(len(rgb_color)):
            labels.append(matplotlib.lines.Line2D([], [], color=rgb_color[i], linewidth=5, label=label_color[i]))
    if hasEdgeWeightInfo:
        edgeWeightName = info['edgeWeightName']
        edgeWeightValuesFull = info['edgeWeightValues']
        # reduce to 3 values to keep legend small
        edgeWeightValuesMax = np.max(np.array(edgeWeightValuesFull))
        edgeWeightValuesMin = np.min(np.array(edgeWeightValuesFull))
        edgeWeightValues = []
        edgeWeightValues.append(edgeWeightValuesMin)
        edgeWeightValues.append(0.5 * (edgeWeightValuesMax - edgeWeightValuesMin))
        edgeWeightValues.append(edgeWeightValuesMax)
        edgeWeightLineWidths = [width_max, 0.5 * (width_min + width_max), width_min]

        # create legend entry
        label_weight = ["%s = %s" % (edgeWeightName, c) for c in set(edgeWeightValues)]
        for i in range(len(edgeWeightValues)):
            labels.append(matplotlib.lines.Line2D([], [], color='black', linewidth=edgeWeightLineWidths[i], label=label_weight[i]))

    plt.legend(handles=labels)

def getConflictCluster(pointConflicts, parallelConflicts, npmin=2, npmax=10, plot=True):
    """ Calculate the partition of a given graph with maximal cluster coefficient

    Arguments:
        pointConflicts: Pandas Dataframe containing the point conflicts
        parallelConflicts: Pandas Dataframe containing the parallel conflicts
        npmin: Minimum number of partitions to search for
        npmax: Maximum number of partitions to search for
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
    for nparts in range(npmin, npmax + 1):
        p = metis.part_graph(G, nparts=nparts)
        graphs = getPartitions(G, p[1])
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
        nx.draw(G, node_size=300, pos=layout, node_color=partition_color)

    return maxClusterPartitioning[1], maxClusterPartition

def main():
    parser = argparse.ArgumentParser(description='Calculate point conflicts from trajectory data', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(help="Give a keyword", dest='mode')
    parser.add_argument('--input', default='data/TrajDataV2_20120729.txt', help='input file containing the trajectory data with consecutive flight index')
    parser.add_argument('-d', '--mindistance', default=30, help='Minimum distance in nautic miles to qualify as a conflict', type=float)
    parser.add_argument('-t', '--mintime', default=60, help='Minimum time difference in minutes to qualify as a conflict', type=int)
    parser.add_argument('--pointConflictFile', help='input file containing the point conflicts (overwrites -t and -d)')
    parser.add_argument('--parallelConflictFile', help='input file containing the parallel conflicts (overwrites -t and -d)')
    parser.add_argument('--multiConflictFile', help='input file containing the conflicts between pairwise conflicts (overwrites -t and -d)')
    parser.add_argument('--flights2ConflictsFile', help='input file the mapping from flight to conflict indices (overwrites -t and -d)')
    parser.add_argument('--rawPointConflictFile', help='input file containing the raw point conflicts')

    all_parser = subparsers.add_parser("all", help='Plot all trajectories and raw point conflicts', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    all_parser.add_argument('--eastwest', action='store_true', help='plot eastbound and westbound flights in different colors')

    conflict_parser = subparsers.add_parser("conflict", help='Plot a special conflicts', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    conflict_parser.add_argument('--info', action='store_true', help='Show info for all conflicts without plotting')
    conflict_parser.add_argument('-k', '--conflictIndex', default=0, help='Conflict index to plot', type=int)

    flight_parser = subparsers.add_parser("flight", help='Plot a special flight including all conflicts', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    flight_parser.add_argument('-i', '--flightIndex', default=None, help='flight index to plot', type=int)
    flight_parser.add_argument('--nfmin', default=0, help='minimal index in the list of flights ordered by their conflicts to include', type=int)
    flight_parser.add_argument('--nfmax', default=1, help='maximal index in the list of flights ordered by their conflicts to include', type=int)

    graph_parser = subparsers.add_parser("graph", help='Plot a conflicting flights as graph', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    graph_parser.add_argument('-n', '--nparts', default=None, help='Number of partitions to plot', type=int)
    graph_parser.add_argument('--multi', action='store_true', help='Plot conflicts between pairwise conflicts instead of pairwise conflicts only')
    group_graph = graph_parser.add_mutually_exclusive_group()
    group_graph.add_argument('-p', '--partition', default=None, help='Plot whole graph and highlight a special partition', type=int)
    group_graph.add_argument('--grid', action='store_true', help='Plot several subplots, each highlighting another partition in the whole graph')
    group_graph.add_argument('--separate', action='store_true', help='Spatially separate partitions in plot')
    group_graph.add_argument('--component', action='store_true', help='Plot all connected components of the graph')

    subset_parser = subparsers.add_parser("subset", help='Calculate disjunct subset with maximal internal cluster coefficient', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subset_parser.add_argument('-n', '--npmin', default=2, help='Minimal number of partitions to search', type=int)
    subset_parser.add_argument('-m', '--npmax', default=10, help='Maximal number of partitions to search', type=int)

    args = parser.parse_args()

    trajectoryFile = '%s.csv' % args.input
    trajectories = pd.read_csv(trajectoryFile, index_col='flightIndex')
    name = "mindist%05.1f_mintime%03i" % (args.mindistance, args.mintime)
    rawPointConflictFile = '%s.%s.rawPointConflicts.csv' % (args.input, name)
    pointConflictFile = '%s.%s.pointConflicts.csv' % (args.input, name)
    parallelConflictFile = '%s.%s.parallelConflicts.csv' % (args.input, name)
    multiConflictFile = '%s.%s.multiConflicts.csv' % (args.input, name)
    flights2ConflictsFile = '%s.%s.flights2Conflicts.h5' % (args.input, name)
    if args.rawPointConflictFile:
        rawPointConflictFile = args.rawPointConflictFile
    if args.pointConflictFile:
        pointConflictFile = args.pointConflictFile
    if args.parallelConflictFile:
        parallelConflictFile = args.parallelConflictFile
    if args.multiConflictFile:
        multiConflictFile = args.multiConflictFile
    if args.flights2ConflictsFile:
        flights2Conflicts = args.flights2ConflictsFile

    if args.mode == 'all':
        rawpointConflicts = pd.read_csv(rawPointConflictFile)
        map = prepareWorldMapPlot()
        addTrajectories(map, trajectories, eastWest=args.eastwest)
        addPointConflicts(map, rawpointConflicts)
        plt.show()

    if args.mode == 'conflict':
        pointConflicts = pd.read_csv(pointConflictFile, index_col='conflictIndex')
        parallelConflicts = pd.read_csv(parallelConflictFile, index_col='parallelConflict')
        NPointConflicts = pointConflicts.index.max()
        NParallelConflicts = parallelConflicts.index.max()
        if args.info:
            print "Read point conflicts from ", pointConflictFile
            print "Read paraellel conflicts from", parallelConflictFile
            print "Point conflict indices range from 0 to", NPointConflicts - 1
            print "Parallel conflict indices range from", NPointConflicts, " to", NParallelConflicts
        else:
            plotConflict(args.conflictIndex, trajectories, pointConflicts, parallelConflicts)
            plt.show()

    if args.mode == 'flight':
        pointConflicts = pd.read_csv(pointConflictFile, index_col='conflictIndex')
        parallelConflicts = pd.read_csv(parallelConflictFile, index_col='parallelConflict')
        flights2Conflicts = pd.read_hdf(flights2ConflictsFile, 'flights2Conflicts')
        map = prepareWorldMapPlot()
        if args.flightIndex:
            addFlightsAndConflicts(map, [args.flightIndex], trajectories, pointConflicts, parallelConflicts, flights2Conflicts)
        else:
            addMostInvolvedFlightsAndConflicts(map, args.nfmin, args.nfmax, trajectories, pointConflicts, parallelConflicts, flights2Conflicts)
        plt.show()
    if args.mode == 'graph':
        if not args.multi:
            pointConflicts = pd.read_csv(pointConflictFile, index_col='conflictIndex')
            parallelConflicts = pd.read_csv(parallelConflictFile, index_col='parallelConflict')
            G = getConflictGraph(pointConflicts, parallelConflicts)
            plotGraph(G, nparts=args.nparts, partition=args.partition, separate=args.separate, grid=args.grid, connectedComponents=args.component)
        else:
            pointConflicts = pd.read_csv(pointConflictFile, index_col='conflictIndex')
            multiConflicts = pd.read_csv(multiConflictFile, index_col='multiConflictIndex')
            G = getMultiConflictGraph(multiConflicts)
            plotGraph(G, nparts=args.nparts, partition=args.partition, separate=args.separate, grid=args.grid, connectedComponents=args.component)
        plt.show()

    if args.mode == 'subset':
        pointConflicts = pd.read_csv(pointConflictFile, index_col='conflictIndex')
        parallelConflicts = pd.read_csv(parallelConflictFile, index_col='parallelConflict')
        partitioning, partition = getConflictCluster(pointConflicts, parallelConflicts, npmin=args.npmin, npmax=args.npmax, plot=True)
        plt.show()

if __name__ == "__main__":
    main()
