Preprocessing for the ATM data
==============================

Code
----

The code can be found in the repository under

    /precalculation/

Prerequisites
-------------

The code was tested with the following python libraries:

 - matplotlib 1.5.1
 - pandas 0.18.1
 - numpy 1.11.0
 - argparse 1.1
 - networkx 1.11
 - cython 0.23.4
 - progressbar 2.3



Data processing
---------------
The purpose of this code is to read in the raw ATM trajectory data and process it for the mapping to a QUBO.
This is done by the

    ./atm.py --multi

command and involves the following steps:

##### 1. Read in the raw trajectory data and introduce new flight index #####
 - Input: ``data/TrajDataV2_20120729.txt``
 - Output: ``data/TrajDataV2_20120729.txt.csv``

##### 2. Detect raw point conflicts #####
 - Input: ``data/TrajDataV2_20120729.txt.csv``
 - Parameter: 
   1. mindist: Threshold for space in nautical miles
   2. mintime: Threshold for time in minutes

 - Output: 
   ``data/TrajDataV2_20120729.txt.mindist030.0_mintime060.rawPointConflicts.csv`` 
   - Pandas DataFrame containing the raw point conflicts with columns
     1. conflictIndex
     2. flight1
     3. flight2
     4. latitude1
     5. latitude2
     6. longitude1
     7. longitude2
     8. alt1
     9. alt2
     10. time1
     11. time2

##### 3. Categorize the conflicts into point conflicts and parallel conflicts ####
 - Output 1: 
   ``data/TrajDataV2_20120729.txt.mindist030.0_mintime060.pointConflicts.csv``
   - Pandas DataFrame with columns:
     1. conflictIndex
     2. flight1
     3. flight2
     4. latitude1
     5. latitude2
     6. longitude1
     7. longitude2
     8. time1
     9. time2
     10. alt1
     11. alt2

 - Output 2: 
   ``data/TrajDataV2_20120729.txt.mindist030.0_mintime060.parallelConflicts.csv`` 
   - Pandas DataFrame with columns:
     1. parallelConflict index
     2. flight1
     3. flight2
     4. latitude1
     5. latitude2
     6. longitude1
     7. longitude2
     8. time1
     9. time2
     10. alt1
     11. alt2

##### 4. Drop all conflicts which can not become real by a self-consistent algorithm #####
 - Parameter: 
   1. delayPerConflictAvoidance: Delay introduced by each conflict avoiding maneuver
   2. dthreshold: temporal threshold below which a conflict is considered real
   3. maxDepartDelay: maximal delay at departure time

 - Output 1: 
   ``data/TrajDataV2_20120729.txt.mindist030.0_mintime060.reducedPointConflicts_delay003_thres003_depart010.csv``
   - Pandas DataFrame with columns:
     1. conflictIndex
     2. flight1
     3. flight2
     4. latitude1
     5. latitude2
     6. longitude1
     7. longitude2
     8. time1
     9. time2
     10. alt1
     11. alt2

 - Output 2: 
   ``data/TrajDataV2_20120729.txt.mindist030.0_mintime060.reducedParallelConflicts_delay003_thres003_depart010.csv``
   - Pandas DataFrame with columns:
     1. parallelConflict index
     2. flight1
     3. flight2
     4. latitude1
     5. latitude2
     6. longitude1
     7. longitude2
     8. time1
     9. time2
     10. alt1
     11. alt2

##### 5. Calculate the mapping from flight index to conflict index. I.e. for each flight a list of conflicts the flight is involved in. ####
 - Output: 
   ``data/TrajDataV2_20120729.txt.mindist030.0_mintime060.flights2Conflicts_delay003_thres003_depart010.h5``
   - Pandas panel (array of DataFrames) containing the mapping from the flight index
     to the conflicts (in temporal order)
     first dimension: flight indices
     second and third dimension: Pandas DataFrame with columns
     1. arrival time
     2. arrival time of the partner flight
     3. partner flight
     4. consecutive conflict index


#### 6. Calculate the non-pairwise conflicts #####
 - Output: 
   ``data/TrajDataV2_20120729.txt.mindist030.0_mintime060.multiConflicts_delay003_thres003_depart010.csv``
   - Pandas DataFrame with columns
     1. multiConflictIndex : consecutive index for the non-pairwise conflicts
     2. conflict1: first conflict index
     3. conflict2: second conflict index
     4. conflictType1: True for parallel conflict, false for point conflict
     5. conflictType2: True for parallel conflict, false for point conflict
     6. isConflict11: True if there is a point conflict between the 1st flight of the 1st conflict and the 1st flight of the 2nd conflict
     7. isConflict12: True if there is a point conflict between the 1st flight of the 1st conflict and the 2nd flight of the 2nd conflict
     8. isConflict21: True if there is a point conflict between the 2nd flight of the 1st conflict and the 1st flight of the 2nd conflict
     9. isConflict22: True if there is a point conflict between the 2nd flight of the 1st conflict and the 2nd flight of the 2nd conflict
     10. deltaTMin': Minimum time difference of all contributing point conflicts
     11. multiConflictType': 0 if both conflicts are point conflicts, 1 if both conflicts are parallel conflicts, 0.5 otherwise

Further details:

    ./atm.py -h

and [/talks/20160726_Stollenwerk_QuAASI_Juelich/20160726_QuAASI_ATM.pdf](https://babelfish.arc.nasa.gov/trac/qcfatm/export/022204deb39998241c11c8f404c74e31f05a9703/talks/20160726_Stollenwerk_QuAASI_Juelich/20160726_QuAASI_ATM.pdf)


Data analysis
-------------

The tool for analysing the data can be invoked by

   ./analysis.py <keyword> <options>

See

    ./analysis.py -h

for allowed keywords and

    ./analysis.py <keyword> -h

for information about the usage.
