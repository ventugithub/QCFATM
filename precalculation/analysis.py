#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.lines
import pandas as pd
import numpy as np
import tools
import argparse
import networkx as nx

def prepareWorldMapPlot(llcrnrlon=None, llcrnrlat=None, urcrnrlon=None, urcrnrlat=None, centerLat=0.0, centerLon=0.0):
    # Create a figure of size (i.e. pretty big)
    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(1, 1, 1)

    # Create a map, using the Gall-Peters projection,
    from mpl_toolkits.basemap import Basemap
    worldmap = Basemap(projection='gall',
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
    worldmap.drawcoastlines()

    # Draw country borders on the map
    worldmap.drawcountries()

    # Fill the land with grey
    worldmap.fillcontinents(color='#888888')

    # Draw the map boundaries
    worldmap.drawmapboundary(fill_color='#f4f4f4')
    return worldmap

def addPoints(worldmap, trajectory, markersize=2, color='b', marker='+', linewidth=1, linestyle='-', latitude='latitude', longitude='longitude'):
    """ Plot trajectory points

    Arguments:
        worldmap: basemap object for plotting
        trajectory: Pandas object containing columns for latitude and longitude
        markersize: matplotlib markersize
        marker: matplotlib marker
        linewidth: matplotlib linewidth
        linestyle: matplotlib linestyle
        latitude: name of the latitude column
        longitude: name of the longitude column
    """
    x, y = worldmap(np.array(trajectory[longitude]), np.array(trajectory[latitude]))
    worldmap.plot(x, y, 'b', color=color, linestyle=linestyle, linewidth=linewidth, markersize=markersize, marker=marker)

def addTrajectories(worldmap, trajectories, eastWest=False):
    if eastWest:
        for flightIndex in set(trajectories.index):
            x, y = worldmap(trajectories[trajectories.index == flightIndex]['longitude'].values, trajectories[trajectories.index == flightIndex]['latitude'].values)
            if x[1] > x[0]:
                worldmap.plot(x, y, 'b', markersize=2, marker='+')
            else:
                worldmap.plot(x, y, color='brown', markersize=2, marker='+')
    else:
        x, y = worldmap(trajectories['longitude'].values, trajectories['latitude'].values)
        worldmap.plot(x, y, 'b', markersize=2, marker='+')


def addPointConflicts(worldmap, pointConflicts):
    x, y = worldmap(pointConflicts['lon1'].values, pointConflicts['lat1'].values)
    worldmap.plot(x, y, 'r', markersize=6, marker='<', linestyle='None')
    x, y = worldmap(pointConflicts['lon2'].values, pointConflicts['lat2'].values)
    worldmap.plot(x, y, 'g', markersize=6, marker='>', linestyle='None')

def addConflictPlot(worldmap, conflictIndex, trajectories, pointConflicts, parallelConflicts, red=False):
    """ Given a conflict index, plot the trajectories of the involved flights and the conflicting trajectory points

    Arguments:
        worldmap: basemap object for plotting
        conflictIndex: conflict index
        trajectories: Pandas Dataframe containing all trajectories
        pointConflicts: Pandas Dataframe containing the point conflicts
        parallelConflicts: Pandas Dataframe containing the parallel conflicts
        red: plot all conflict points in red (default false)
    """
    # plot involved flight trajectories
    flight1, flight2, conflictTrajectoryPoints = tools.getInvolvedFlights(conflictIndex, pointConflicts, parallelConflicts)

    addPoints(worldmap, trajectories.loc[flight1], markersize=4, marker='+')
    addPoints(worldmap, trajectories.loc[flight2], markersize=4, marker='+')
    # point conflict
    col = 'r' if red else 'g'
    addPoints(worldmap, conflictTrajectoryPoints, color=col, markersize=6, linewidth=6, marker='o', linestyle='-', latitude='lat1', longitude='lon1')
    addPoints(worldmap, conflictTrajectoryPoints, color='r', markersize=6, linewidth=6, marker='o', linestyle='-', latitude='lat2', longitude='lon2')

def plotConflicts(conflictIndices, trajectories, pointConflicts, parallelConflicts, npoints=3, ax=None, verbose=False):
    """ Given a conflict index, plot the trajectories of the involved flights and the conflicting trajectory points
    around the conflict region

    Arguments:
        conflictIndices: sequence of conflict indices
        trajectories: Pandas Dataframe containing all trajectories
        pointConflicts: Pandas Dataframe containing the point conflicts
        parallelConflicts: Pandas Dataframe containing the parallel conflicts
        npoints: number of trajectorie points before and after the conflicts to be plotted
        ax: matplotlib axes object (optional)
    """
    flights1 = []
    flights2 = []
    conflictTrajectoryPointsContainer = []
    mintimes1 = []
    maxtimes1 = []
    mintimes2 = []
    maxtimes2 = []
    centerLons = []
    centerLats = []
    for conflictIndex in conflictIndices:
        flight1, flight2, conflictTrajectoryPoints = tools.getInvolvedFlights(conflictIndex, pointConflicts, parallelConflicts)
        if isinstance(conflictTrajectoryPoints, pd.core.series.Series):
            mintime1 = conflictTrajectoryPoints.time1
            maxtime1 = conflictTrajectoryPoints.time1
            mintime2 = conflictTrajectoryPoints.time2
            maxtime2 = conflictTrajectoryPoints.time2
            centerLon = 0.5 * (conflictTrajectoryPoints.lon1 + conflictTrajectoryPoints.lon2)
            centerLat = 0.5 * (conflictTrajectoryPoints.lat1 + conflictTrajectoryPoints.lat2)

        elif isinstance(conflictTrajectoryPoints, pd.core.frame.DataFrame):
            mintime1 = conflictTrajectoryPoints.time1.min()
            maxtime1 = conflictTrajectoryPoints.time1.max()
            mintime2 = conflictTrajectoryPoints.time2.min()
            maxtime2 = conflictTrajectoryPoints.time2.max()
            centerLon = 0.5 * (conflictTrajectoryPoints.lon1.mean() + conflictTrajectoryPoints.lon2.mean())
            centerLat = 0.5 * (conflictTrajectoryPoints.lat1.mean() + conflictTrajectoryPoints.lat2.mean())

        else:
            raise ValueError('plotConflict: conflictTrajectoryPoints is neither pandas series nor dataframe')
        flights1.append(flight1)
        flights2.append(flight2)
        conflictTrajectoryPointsContainer.append(conflictTrajectoryPoints)
        mintimes1.append(mintime1)
        maxtimes1.append(maxtime1)
        mintimes2.append(mintime2)
        maxtimes2.append(maxtime2)
        centerLons.append(centerLon)
        centerLats.append(centerLat)

    if len(conflictIndices) > 1:
        centerLon = np.array(centerLons).mean()
        centerLat = np.array(centerLats).mean()

    # Create a figure
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    t1 = [] * len(conflictIndices)
    t2 = [] * len(conflictIndices)
    t = trajectories.reset_index()
    for i in range(len(conflictIndices)):
        flight1 = flights1[i]
        flight2 = flights2[i]
        conflictTrajectoryPoints = conflictTrajectoryPointsContainer[i]

        # plot involved flight trajectories, flight 1
        minindex = t[(t.flightIndex == flight1) & (t.time == mintimes1[i])].index.values[0]
        maxindex = t[(t.flightIndex == flight1) & (t.time == maxtimes1[i])].index.values[0]
        subset = t[t.flightIndex == flight1]
        minindex = max(subset.index[0], minindex - npoints)
        maxindex = min(subset.index[-1], maxindex + npoints)
        t1.append(subset.loc[minindex:maxindex])

        # plot involved flight trajectories, flight 2
        minindex = t[(t.flightIndex == flight2) & (t.time == mintimes2[i])].index.values[0]
        maxindex = t[(t.flightIndex == flight2) & (t.time == maxtimes2[i])].index.values[0]
        subset = t[t.flightIndex == flight2]
        minindex = max(subset.index[0], minindex - npoints)
        maxindex = min(subset.index[-1], maxindex + npoints)
        t2.append(subset.loc[minindex:maxindex])

    from mpl_toolkits.basemap import Basemap
    worldmap = Basemap(projection='gall',
                       # with low resolution,
                       resolution='l',
                       # And threshold 100000
                       area_thresh=100000.0,
                       # Center
                       lat_0=centerLat, lon_0=centerLon)
    for i in range(len(conflictIndices)):
        x, y = worldmap(np.array(t1[i]['longitude']), np.array(t1[i]['latitude']))
        for k in range(1, len(x)):
            ax.annotate("",
                        xytext=(x[k - 1], y[k - 1]), xycoords='data',
                        xy=(x[k], y[k]), textcoords='data',
                        arrowprops=dict(arrowstyle="-|>", color='b', connectionstyle="arc3"),
                        )
        ax.plot(x, y, color='b', linestyle='-')
        x, y = worldmap(np.array(t2[i]['longitude']), np.array(t2[i]['latitude']))
        for k in range(1, len(x)):
            ax.annotate("",
                        xytext=(x[k - 1], y[k - 1]), xycoords='data',
                        # xy=(xmid, ymid), textcoords='data',
                        xy=(x[k], y[k]), textcoords='data',
                        arrowprops=dict(arrowstyle="-|>", color='g', connectionstyle="arc3"),
                        )

        ax.plot(x, y, color='g', linestyle='-')
        # point conflict
        x, y = worldmap(np.array(conflictTrajectoryPoints['lon1']), np.array(conflictTrajectoryPoints['lat1']))
        ax.plot(x, y, color='r',  markersize=5, marker='o')
        x, y = worldmap(np.array(conflictTrajectoryPoints['lon2']), np.array(conflictTrajectoryPoints['lat2']))
        ax.plot(x, y, color='violet',  markersize=5, marker='o')

        if verbose:
            lon1 = t1[i]['longitude'].values
            lon2 = t2[i]['longitude'].values
            lat1 = t1[i]['latitude'].values
            lat2 = t2[i]['latitude'].values
            time1 = t1[i]['time'].values
            time2 = t2[i]['time'].values
            CosD = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * (np.cos(lon1) * np.cos(lon2) + np.sin(lon1) * np.sin(lon2))
            CosD = np.minimum(CosD, np.ones_like(CosD))
            CosD = np.maximum(CosD, -np.ones_like(CosD))
            earthRadius = 6367.0
            spatialDistance = earthRadius * np.arccos(CosD)
            temporatDistance = time1 - time2
            print "  lon1   lon2   lat1   lat2 spatialDistance(km) temporalDistance(min)"
            for lo1, lo2, la1, la2, ds, dt in zip(lon1, lon2, lat1, lat2, spatialDistance, temporatDistance):
                print "%+6.2f %+6.2f %+6.2f %+6.2f %+19.2f %+21.2f" % (lo1, lo2, la1, la2, ds, dt)

    if len(conflictIndices) == 1:
        ax.set_title("$k=%i, f_1=%i, f_2=%i$" % (conflictIndices[0], flights1[0], flights2[0]))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_autoscale_on(True)


def addFlightsAndConflicts(worldmap, flightIndices, trajectories, pointConflicts, parallelConflicts, flights2Conflicts, blue=False, red=False):
    """ Given a flight index, plot the trajectories of the involved flights and the conflicting trajectory points

    Arguments:
        worldmap: basemap object for plotting
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
            addConflictPlot(worldmap, conflictIndex, trajectories, pointConflicts, parallelConflicts, red=red)
        addPoints(worldmap, trajectories.loc[flightIndex], color=col, markersize=6, marker='+')

def addMostInvolvedFlightsAndConflicts(worldmap, nfmin, nfmax, trajectories, pointConflicts, parallelConflicts, flights2Conflicts):
    """ Plot the trajectories and conflicts of flights with the highest number of conflicts

    Arguments:
        worldmap: basemap object for plotting
        nfmin: minimal index in the list of flights ordered by their conflicts to include
        nfmax: maximal index in the list of flights ordered by their conflicts to include
        trajectories: Pandas Dataframe containing all trajectories
        pointConflicts: Pandas Dataframe containing the point conflicts
        parallelConflicts: Pandas Dataframe containing the parallel conflicts
        flights2Conflicts: Pandas panel containing the mapping from flight index to conflict indices
    """
    # get the flights with the most number of conflicts
    mostInvolvedFlights = flights2Conflicts.count().T.sort_values('conflictIndex', ascending=False).index[nfmin:nfmax + 1].values
    addFlightsAndConflicts(worldmap, mostInvolvedFlights, trajectories, pointConflicts, parallelConflicts, flights2Conflicts, blue=True, red=True)

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
    parser.add_argument('-t', '--mintime', default=18, help='Minimum time difference in minutes to qualify as a potential conflict', type=int)
    parser.add_argument('--delayPerConflict', default=0, help='Delay introduced by each conflict avoiding maneuver', type=int)
    parser.add_argument('--dthreshold', default=3, help='Minimum time difference in minutes to qualify as a real conflict', type=int)
    parser.add_argument('--maxDepartDelay', default=18, help='Maximum departure delay', type=int)
    parser.add_argument('--pointConflictFile', help='input file containing the point conflicts (overwrites -t and -d)')
    parser.add_argument('--parallelConflictFile', help='input file containing the parallel conflicts (overwrites -t and -d)')
    parser.add_argument('--multiConflictFile', help='input file containing the conflicts between pairwise conflicts (overwrites -t and -d)')
    parser.add_argument('--flights2ConflictsFile', help='input file the mapping from flight to conflict indices (overwrites -t and -d)')
    parser.add_argument('--rawPointConflictFile', help='input file containing the raw point conflicts')

    all_parser = subparsers.add_parser("all", help='Plot all trajectories and raw point conflicts', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    all_parser.add_argument('--eastwest', action='store_true', help='plot eastbound and westbound flights in different colors')

    conflict_parser = subparsers.add_parser("conflict", help='Plot a special conflicts', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    conflict_parser.add_argument('--info', action='store_true', help='Show info for all conflicts without plotting')
    conflict_parser.add_argument('-k', '--conflictIndices', nargs='+', help='Conflict index to plot', type=int)
    conflict_parser.add_argument('--allpoints', action='store_true', help='Plot all trajectory points')
    conflict_parser.add_argument('--verbose', action='store_true', help='Show coordinates and distances')

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
    flights2ConflictsFile = "%s.%s.flights2Conflicts_delay%03i_thres%03i_depart%03i.h5" % (args.input, name, args.delayPerConflict, args.dthreshold, args.maxDepartDelay)
    pointConflictFile = "%s.%s.reducedPointConflicts_delay%03i_thres%03i_depart%03i.csv" % (args.input, name, args.delayPerConflict, args.dthreshold, args.maxDepartDelay)
    parallelConflictFile = "%s.%s.reducedParallelConflicts_delay%03i_thres%03i_depart%03i.csv" % (args.input, name, args.delayPerConflict, args.dthreshold, args.maxDepartDelay)
    multiConflictFile = "%s.%s.multiConflicts_delay%03i_thres%03i_depart%03i.h5" % (args.input, name, args.delayPerConflict, args.dthreshold, args.maxDepartDelay)
    if args.rawPointConflictFile:
        rawPointConflictFile = args.rawPointConflictFile
    if args.pointConflictFile:
        pointConflictFile = args.pointConflictFile
    if args.parallelConflictFile:
        parallelConflictFile = args.parallelConflictFile
    if args.multiConflictFile:
        multiConflictFile = args.multiConflictFile
    if args.flights2ConflictsFile:
        flights2ConflictsFile = args.flights2ConflictsFile

    if args.mode == 'all':
        rawpointConflicts = pd.read_csv(rawPointConflictFile)
        worldmap = prepareWorldMapPlot()
        addTrajectories(worldmap, trajectories, eastWest=args.eastwest)
        addPointConflicts(worldmap, rawpointConflicts)
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
        elif args.allpoints:
            worldmap = prepareWorldMapPlot()
            for conflictIndex in args.conflictIndices:
                addConflictPlot(worldmap, conflictIndex, trajectories, pointConflicts, parallelConflicts)
            plt.show()
        else:
            plotConflicts(args.conflictIndices, trajectories, pointConflicts, parallelConflicts, verbose=args.verbose)
            plt.show()

    if args.mode == 'flight':
        pointConflicts = pd.read_csv(pointConflictFile, index_col='conflictIndex')
        parallelConflicts = pd.read_csv(parallelConflictFile, index_col='parallelConflict')
        flights2Conflicts = pd.read_hdf(flights2ConflictsFile, 'flights2Conflicts')
        worldmap = prepareWorldMapPlot()
        if args.flightIndex:
            addFlightsAndConflicts(worldmap, [args.flightIndex], trajectories, pointConflicts, parallelConflicts, flights2Conflicts)
        else:
            addMostInvolvedFlightsAndConflicts(worldmap, args.nfmin, args.nfmax, trajectories, pointConflicts, parallelConflicts, flights2Conflicts)
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
