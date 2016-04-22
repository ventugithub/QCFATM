import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import conflict
import os
from mpl_toolkits.basemap import Basemap

def plotTrajectories(trajectories):
    # Create a figure of size (i.e. pretty big)
    plt.figure(figsize=(20, 10))

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
    x, y = map(trajectories['longitude'].values, trajectories['latitude'].values)

    # Plot them using round markers of size 6
    map.plot(x, y, 'b', markersize=6)

    # Show the map
    plt.show()


def distance(lat1, lon1, lat2, lon2, R=6367):
    """Get the distance on a great circle between to trajectory points in kilometers

    Arguments:
    R: Radius in kilometers
    lat1: latitude of first point in degrees
    lon1: longitude of the first point in degrees
    lat2: latitude of the second point in degrees
    lon2: longitude of the second point in degrees
    """
    Lat0 = np.radians(lat1)
    Latf = np.radians(lat2)
    Lon0 = np.radians(lon1)
    Lonf = np.radians(lon2)

    return R * np.arccos(np.sin(Lat0) * np.sin(Latf) + np.cos(Lonf-Lon0)*np.cos(Lat0)*np.cos(Latf))

def detectSpatialConflict(lat1, lon1, lat2, lon2, mindistance):
    return distance(lat1, lon1, lat2, lon2) < mindistance


# constants
# nautic mile in kilometers
nautic = 1.852

# minimal acceptable distance in kilometers
mindistance = 30 * nautic

# minimal acceptable time difference
mintime = 3

inputDataFile = "data/TrajDataV2_20120729.txt"
filename = 'trajectories'
trajectories = None
if not os.path.exists(filename + '.h5'):
    # read in data from July, 29th, 2012
    print "read in trajectories ..."
    trajectories = pd.read_csv(inputDataFile,
                               delimiter="\t",
                               names=('flight', 'date', 'wind', 'time', 'speed', 'altitude', 'latitude', 'longitude', 'nan'),
                               usecols=('flight', 'date', 'wind', 'time', 'speed', 'altitude', 'latitude', 'longitude')
                               )
    print "add consecutive flight index to data ..."
    # add consecutive flight index to the data
    flightNames = trajectories['flight'].unique()
    trajectories['flightIndex'] = trajectories['flight'].map(lambda x: np.where(flightNames == x)[0][0])
    # set consecutive flight index as dataset index
    trajectories = trajectories.set_index('flightIndex')
    trajectories.to_hdf(filename + '.h5', 'trajectories', mode='w')
else:
    print "read in trajectories ..."
    trajectories = pd.read_hdf(filename + '.h5', 'trajectories')

conflict.detectConflicts(trajectories.index, trajectories.time, trajectories.latitude, trajectories.longitude)

# print "detect conflicts ..."
# t = trajectories[trajectories.time == 19]
# flightPairs = np.array(list(itertools.combinations(t.index, 2)))
# df = pd.DataFrame(flightPairs, columns=['first', 'second'])
# df['isConflict'] = df.apply(lambda x:
                            # detectSpatialConflict(t['latitude'][x['first']],
                                                  # t['longitude'][x['first']],
                                                  # t['latitude'][x['second']],
                                                  # t['longitude'][x['second']],
                                                  # mindistance),
                            # axis=1)


# print np.array(trajectories.index).min()
# print np.array(trajectories.index).max()
# print "loop all conflicts ..."
# n = 0
# for (i, j) in itertools.combinations(trajectories.index.unique(), 2):
    # n = n + 1
    # if n % 10000 == 0:
        # print i, j
