#!/usr/bin/env python
import glob
import os
import argparse
import multiprocessing
import filelock
import pandas as pd
import numpy as np
import solveInstance as si

def solveAndCheckValidity(instancefile, w2, w3, **solve_instance_args):
    """
    solve the instance for given penalty weights. gives back validity and energy of solution
    """
    si.solve_instance(instancefile, penalty_weights={'unique': w2, 'conflict': w3}, **solve_instance_args)
    inventory = pd.read_csv(solve_instance_args['inventoryfile'])
    subset = inventory[(inventory.instance == instancefile) & (np.abs(inventory.penalty_weight_unique - w2) < 1E-14) & (np.abs(inventory.penalty_weight_conflict - w3) < 1E-14) & (inventory.exact)]
    if len(subset) == 0:
        print "Warning: No exact solution available"
        return None
    elif len(subset) != 1:
        print "Duplicates in inventory:"
        print inventory
        print subset
        raise ValueError('Duplicates in inventory')

    isValid = subset.iloc[0]['isValid']
    energy = subset.iloc[0]['energy']
    return isValid, energy

def getPointOnCircle(center, radius, angle):
    """
    returns point on circle around a center.
    the angle is measured from the point on the circle with the largest distance to the origin
    """
    l = np.linalg.norm(center)
    if l:
        xunit = center / np.linalg.norm(center)
    else:
        xunit = np.array([1, 0])
    deltax = radius * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]).dot(xunit)
    return center + deltax

def findThresholdOnCircle(wunique, wconflict, radiuses, direction, delta_w, instancefile, inventoryfile_penalty_threshold, validityMapFile, **solve_instance_args):
    """
    Find threshold on circle around a given starting point (wunique, wconflict)
    which is assumed to be valid
    """
    foundInvalid = False
    center = np.array([wunique, wconflict])
    f = open(validityMapFile, 'a')
    # check point nearest to the origin is invalid, if not invalid increase radius
    uinv = None
    vinv = None
    for r in radiuses:
        radius = r
        x = getPointOnCircle(center, radius, np.pi)
        u = max(np.round(x[0], 3), 0.001)
        v = max(np.round(x[1], 3), 0.001)
        uinv = u
        vinv = v
        print "Threshold detection algorithm is at radius %f. Point = (%0.3f, %0.3f)" % (radius, u, v)
        valid, energy = solveAndCheckValidity(instancefile, u, v, **solve_instance_args)
        # store data
        f.write("%0.3f,%0.3f,%i,%e\n" % (u, v, valid, energy))
        # for trivial solutions, any choice of penealy weights yields the result
        if abs(energy) < 1E-13:
            print "WARNING: Energy of solution is zero. Stop search for penalty weight threshold on circle"
            break
        if valid is None or np.isnan(valid):
            print "WARNING: No exact solution available. Stop search for penalty weight threshold on circle"
            break
        if not valid:
            foundInvalid = True
            break
    if not foundInvalid:
        raise ValueError('maximal radius %f seem to be to small' % radius)

    # bisection
    found = False
    angle = 0.25 * np.pi
    angleabove = None
    # store rounded invalid value
    anglebelow = 0
    while not found and angle < 2 * np.pi:
        x = getPointOnCircle(center, radius, np.pi + direction * angle)
        u = max(np.round(x[0], 3), 0.001)
        v = max(np.round(x[1], 3), 0.001)
        print "Threshold detection bisection algorithm is at angle %0.3f Pi. Point = (%0.3f, %0.3f)" % (angle / np.pi, u, v)
        valid, energy = solveAndCheckValidity(instancefile, u, v, **solve_instance_args)
        # store data
        f.write("%0.3f,%0.3f,%i,%e\n" % (u, v, valid, energy))
        # for trivial solutions, any choice of penealy weights yields the result
        if abs(energy) < 1E-13:
            wu = 0
            wc = 0
            found = True
            print "WARNING: Energy of solution is zero. Stop search for penalty weight threshold"
            break
        # break if exact solution was not found
        if valid is None or np.isnan(valid):
            print "WARNING: No exact solution available. Stop search for penalty weight threshold"
            break
        # as long as no valid solution is found increase the angle
        if not valid and not angleabove:
            anglebelow = angle
            uinv = u
            vinv = v
            angle += 0.25 * np.pi
        # if current value of penalty weights yields non valid solution
        elif not valid and angleabove:
            anglebelow = angle
            uinv = u
            vinv = v
            angle = 0.5 * (angleabove + anglebelow)
        # if current value of penalty weights yields valid solution
        else:
            angleabove = angle
            angle = 0.5 * (angleabove + anglebelow)
        # solution is found if the difference between valid and invalid solution < small value
        if valid and abs(u - uinv) < delta_w and abs(v - vinv) < delta_w:
            found = True
            wu = u
            wc = v
    f.close()
    if not found:
        wu = np.nan
        wc = np.nan
    f.close()
    return wu, wc, found

def bisectionToThreshold(wfixedunique, wstart, delta_w, instancefile, inventoryfile_penalty_threshold, validityMapFile, **solve_instance_args):
    """
    given a fixed penalty weight (unique), find the threshold in
    the penalty weight conflict. i.e. the boundary btw. valid and invalid solutions
    """
    w = wstart
    wbelow = wstart
    wabove = None
    found = False
    first = True
    wminabove = None
    f = open(validityMapFile, 'a')
    # bisection to find first validity threshold point
    while not found and w < 1E4:
        w = np.round(w, 3)
        print "Bisection algorithm is at (%0.3f, %0.3f)" % (wfixedunique, w)
        valid, energy = solveAndCheckValidity(instancefile, wfixedunique, w, **solve_instance_args)
        # store data
        f.write("%0.3f,%0.3f,%i,%e\n" % (wfixedunique, w, valid, energy))
        # for trivial solutions, any choice of penealy weights yields the result
        if abs(energy) < 1E-13:
            wminabove = 0
            found = True
            print "WARNING: Energy of solution is zero. Stop search for penalty weight threshold"
            break
        # break if exact solution was not found
        if valid is None or np.isnan(valid):
            print "WARNING: No exact solution available. Stop search for penalty weight threshold"
            break
        # break if first guess is valid
        if first and valid:
            raise ValueError('starting point of bisection is to large: %e' % w)
        # set first flag to false
        first = False
        # as long as no valid solution is found increase the penalty weight
        if not valid and not wabove:
            wbelow = w
            w = w * 10
        # if current value of penalty weights yields non valid solution
        elif not valid and wabove:
            wbelow = w
            w = 0.5 * (wabove + wbelow)
        # if current value of penalty weights yields valid solution
        else:
            wabove = w
            w = 0.5 * (wabove + wbelow)
        # solution is found if the difference between valid and invalid solution < delta_w
        if wabove and abs(wabove - wbelow) < delta_w:
            found = True
            wminabove = wabove
    if not found:
        wminabove = np.nan
    f.close()
    return wfixedunique, wminabove, found

def storeInventory(wunique, wconflict, instancefile, inventoryfile_penalty_threshold):
    """
    Store pairs of penalty weight in inventory
    """
    ivfile = inventoryfile_penalty_threshold
    # use lock file to prevent simultaneous updates to inventory
    # in the case of multiprocessing
    lockfile = ivfile + ".lock"
    lock = filelock.FileLock(lockfile)
    with lock.acquire():
        iv = pd.DataFrame({'instance': [instancefile],
                           'penalty_weight_unique': [wunique],
                           'penalty_weight_conflict': [wconflict]})
        iv = iv.round(3)
        iv.set_index('instance', inplace=True)

        # read in iv file if existent if os.path.exists(ivfile):
        if os.path.exists(ivfile):
            iv_before = pd.read_csv(ivfile, index_col='instance')
            iv = pd.concat([iv_before, iv])
        iv = iv.round(3)

        iv.reset_index(level=0, inplace=True)
        # drop duplicates but ignore version
        iv.drop_duplicates(inplace=True)
        iv.set_index('instance', inplace=True)
        iv.to_csv(ivfile, mode='w')

def solveAndFindPenaltyThreshold(wfixedunique, wstart, delta_w, direction, max_penalty_unique, max_penalty_conflict, radiuses, instancefile, inventoryfile_penalty_threshold, store_inventory, validityMapFile, **solve_instance_args):
    """
    solve instances along the boundary btw. valid and invalid solution.
    """
    # start with bisection for a fixed penalty weight unique
    wunique, wconflict, success = bisectionToThreshold(wfixedunique, wstart, delta_w, instancefile, inventoryfile_penalty_threshold, validityMapFile, **solve_instance_args)
    if success and store_inventory:
        storeInventory(wunique, wconflict, instancefile, inventoryfile_penalty_threshold)
    # radiuses to try to find an invalid point at the point on a circle around the current threshold point with the smallest distance to the origin
    # direction of rotation on a circle around the current threshold point during the search for the next threshold point
    # maximum values of penalty weights
    finished = False
    while not finished:
        wunique, wconflict, success = findThresholdOnCircle(wunique, wconflict, radiuses, direction, delta_w, instancefile, inventoryfile_penalty_threshold, validityMapFile, **solve_instance_args)
        if success and store_inventory:
            print "Found threshold point at (%03f, %03f)" % (wunique, wconflict)
            storeInventory(wunique, wconflict, instancefile, inventoryfile_penalty_threshold)
        elif not success:
            print "WARNING: Search for threshold point on a circle around (%03f, %03f) was not successful." % (wunique, wconflict)
            finished = True
        if wunique > max_penalty_unique or wconflict > max_penalty_conflict:
            print "Threshold point at (%03f, %03f) is out of boundaries. Stop algorithm." % (wunique, wconflict)
            finished = True

def main():
    parser = argparse.ArgumentParser(description='Solve departure only model exactly and scan for threshold in penalty weights at which the solutions become invalid',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--inventoryfile', default='data/instances/analysis/inventory.csv', help='inventory file')
    parser.add_argument('--inventoryfile_penalty_threshold', default='data/instances/analysis/inventory-penalty-weight-threshold.csv', help='inventory file for penalty weight threshold')
    parser.add_argument('--pmin', default=0, help='minimum index of partition to consider', type=int)
    parser.add_argument('--pmax', default=79, help='maximum index of partition to consider', type=int)
    parser.add_argument('--wstart', default=0.001, help='starting value of penalty weight for bisection algorithm (rounded to 3 digits)', type=float)
    parser.add_argument('--delta_w', default=0.01, help='accuray of the bisection algorithm (rounded to 3 digits)', type=float)
    parser.add_argument('-d', '--delays', nargs='+', default=[3], help='delay steps to consider', type=int)
    parser.add_argument('--use_snapshots', action='store_true', help='use snapshot files')
    parser.add_argument('--timeout', default=1000, help='timeout in seconds for exact solver')
    parser.add_argument('--overwrite_validity_map', action='store_true', help='Overwrite valitidy map files')
    parser.add_argument('--fixed_penalty_unique_start', default=2.0, help='fixed penalty weights for unique term of the QUBO for first bisection search of threshold', type=float)
    parser.add_argument('--counter_clockwise', action='store_true', help='Counter clockwise search for threshold points on a circle during the algorithm following the threshold boundary')
    parser.add_argument('--max_penalty_unique', default=2.5, help='maximum penalty weights for unique term of the QUBO to consider during search for threshold', type=float)
    parser.add_argument('--max_penalty_conflict', default=2.5, help='maximum penalty weights for conflict term of the QUBO to consider during search for threshold', type=float)
    parser.add_argument('--radiuses', nargs='+', default=[0.1, 0.2, 0.5, 1], help='radiuses used in algorithm following the threshold boundary', type=float)
    parser.add_argument('--np', default=1, help='number of processes', type=int)
    args = parser.parse_args()

    delays = args.delays
    inventoryfile = args.inventoryfile
    partitions = range(args.pmin, args.pmax + 1)
    timeout = args.timeout
    wstart = args.wstart
    delta_w = args.delta_w
    nproc = args.np
    wfixedunique = args.fixed_penalty_unique_start
    direction = 1 if args.counter_clockwise else -1
    max_penalty_unique = args.max_penalty_unique
    max_penalty_conflict = args.max_penalty_conflict
    radiuses = args.radiuses

    solve_instance_args = {'num_embed': 0,
                           'use_snapshots': True,
                           'unary': True,
                           'verbose': False,
                           'timeout': timeout,
                           'exact': True,
                           'retry_exact': False,
                           'inventoryfile': inventoryfile}

    print "Collect instance files ..."
    instancefiles = []
    for d in delays:
        for p in partitions:
            files = glob.glob('data/instances/instances_d%i/atm_instance_partition%04i_f????_c?????.yaml' % (d, p))
            assert len(files) == 1
            instancefiles.append(files[0])
    print "Solve instances ..."
    if nproc != 1:
        pool = multiprocessing.Pool(processes=nproc)
    for instancefile in instancefiles:
        validityMapFile = instancefile + ".validityMap.csv"
        if args.overwrite_validity_map and os.path.exists(validityMapFile):
            os.remove(validityMapFile)
        if not os.path.exists(validityMapFile):
            f = open(validityMapFile, 'w')
            f.write("penalty_weights_unique,penalty_weights_conflict,validity,energy\n")
            f.close()
        solveAndFindPenaltyThresholdArgs = {'wfixedunique': wfixedunique,
                                            'wstart': wstart,
                                            'delta_w': delta_w,
                                            'direction': direction,
                                            'max_penalty_unique': max_penalty_unique,
                                            'max_penalty_conflict': max_penalty_conflict,
                                            'radiuses': radiuses,
                                            'instancefile': instancefile,
                                            'inventoryfile_penalty_threshold': args.inventoryfile_penalty_threshold,
                                            'store_inventory': True,
                                            'validityMapFile': validityMapFile}
        solveAndFindPenaltyThresholdArgs.update(solve_instance_args)
        if nproc != 1:
            pool.apply_async(solveAndFindPenaltyThreshold, kwds=solveAndFindPenaltyThresholdArgs)
        else:
            solveAndFindPenaltyThreshold(**solveAndFindPenaltyThresholdArgs)

    if nproc != 1:
        pool.close()
        pool.join()

if __name__ == "__main__":
    main()
