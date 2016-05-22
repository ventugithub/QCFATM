import pandas as pd
import numpy as np

def getInvolvedFlights(conflictIndex, pointConflicts, parallelConflicts):
    """ given a conflict index, get both flights involved in the conflict

    Arguments:
        conflictIndex
        pointConflicts
        parallelConflicts

    Returns: flight1, flight2
    """
    NPointConflicts = len(pointConflicts)
    NParallelConflicts = len(parallelConflicts.index)
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
        msg = "Conflict index %i out of range\n" % k
        msg = msg + "Point conflict indices range from 0 to %i\n" % (NPointConflicts - 1)
        msg = msg + "Parallel conflict indices range from %i to %i\n" % (NPointConflicts, NConflicts)
        raise ValueError(msg)

def getInvolvedConflicts(flights2Conflicts, flight):
    """ Given a flight index, return the list of conflicts

    Arguments:
        flights2Conflicts: Pandas panel containing the mapping from flight index to conflict indices
        flight: flight index
    returns:
        a list of conflict indices
    """
    return flights2Conflicts.loc[flight].dropna().sort_values('arrivalTime')['conflictIndex'].values.astype(int)
