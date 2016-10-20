#!/usr/bin/env python
import glob
import os
import argparse
import multiprocessing
import pandas as pd
import numpy as np
import solveInstance as si

def solveAndCheckValidity(instancefile, w2, w3, **solve_instance_args):
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

def solveAndFindPenaltyThreshold(wfixed, wstart, delta_w, instancefile, inventoryfile_penalty_threshold, penalty_weights_unique, penalty_weights_conflict, store_inventory, **solve_instance_args):
    ivfile = inventoryfile_penalty_threshold
    w = wstart
    wbelow = wstart
    wabove = None
    found = False
    first = True
    wminabove = None
    while not found and w < 1E4:
        w = np.round(w, 3)
        print "Bisection algorithm is at", w
        if penalty_weights_unique is not None:
            valid, energy = solveAndCheckValidity(instancefile, wfixed, w, **solve_instance_args)
        else:
            valid, energy = solveAndCheckValidity(instancefile, w, wfixed, **solve_instance_args)
        # for trivial solutions, any choice of penealy weights yields the result
        if energy == 0:
            wminabove = 0
            found = True
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

    if store_inventory:
        if penalty_weights_unique is not None:
            iv = pd.DataFrame({'instance': [instancefile],
                               'penalty_weight_unique': [wfixed],
                               'penalty_weight_conflict': [wminabove],
                               'fixed': ['unique'],
                               })
        else:
            iv = pd.DataFrame({'instance': [instancefile],
                               'penalty_weight_unique': [wminabove],
                               'penalty_weight_conflict': [wfixed],
                               'fixed': ['conflict'],
                               })
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

def main():
    parser = argparse.ArgumentParser(description='Solve departure only model exactly and scan for threshold in penalty weights at which the solutions become invalid')
    parser.add_argument('--inventoryfile', default='data/instances/analysis/inventory.csv', help='inventory file')
    parser.add_argument('--inventoryfile_penalty_threshold', default='data/instances/analysis/inventory-penalty-weight-threshold.csv', help='inventory file for penalty weight threshold')
    parser.add_argument('--pmin', default=0, help='minimum index of partition to consider', type=int)
    parser.add_argument('--pmax', default=79, help='maximum index of partition to consider', type=int)
    parser.add_argument('--wstart', default=0.001, help='starting value of penalty weight for bisection algorithm (rounded to 3 digits)', type=float)
    parser.add_argument('--delta_w', default=0.01, help='accuray of the bisection algorithm (rounded to 3 digits)', type=float)
    parser.add_argument('-d', '--delays', nargs='+', default=[3], help='delay steps to consider', type=int)
    parser.add_argument('--use_snapshots', action='store_true', help='use snapshot files')
    parser.add_argument('--timeout', default=1000, help='timeout in seconds for exact solver')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--fixed_penalty_unique', nargs='+', help='list of fixed penalty weights for unique term of the QUBO', type=float)
    group.add_argument('--fixed_penalty_conflict', nargs='+', help='list of fixed penalty weights for conflict term of the QUBO', type=float)
    parser.add_argument('--np', default=1, help='number of processes', type=int)
    args = parser.parse_args()

    delays = args.delays
    inventoryfile = args.inventoryfile
    partitions = range(args.pmin, args.pmax + 1)
    timeout = args.timeout
    wstart = args.wstart
    delta_w = args.delta_w
    penalty_weights_unique = args.fixed_penalty_unique
    penalty_weights_conflict = args.fixed_penalty_conflict
    nproc = args.np
    if penalty_weights_unique is not None:
        penalty_weights_fixed = penalty_weights_unique
    else:
        penalty_weights_fixed = penalty_weights_conflict

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
        for wfixed in penalty_weights_fixed:
            solveAndFindPenaltyThresholdArgs = {'wfixed': wfixed,
                                                'wstart': wstart,
                                                'delta_w': delta_w,
                                                'instancefile': instancefile,
                                                'inventoryfile_penalty_threshold': args.inventoryfile_penalty_threshold,
                                                'penalty_weights_unique': penalty_weights_unique,
                                                'penalty_weights_conflict': penalty_weights_conflict}
            solveAndFindPenaltyThresholdArgs.update(solve_instance_args)
            if nproc != 1:
                solveAndFindPenaltyThresholdArgs.update({'store_inventory': False})
                pool.apply_async(solveAndFindPenaltyThreshold, kwds=solveAndFindPenaltyThresholdArgs)
            else:
                solveAndFindPenaltyThresholdArgs.update({'store_inventory': True})
                solveAndFindPenaltyThreshold(**solveAndFindPenaltyThresholdArgs)

    if nproc != 1:
        pool.close()
        pool.join()

    if nproc != 1:
        for instancefile in instancefiles:
            for wfixed in penalty_weights_fixed:
                solveAndFindPenaltyThresholdArgs = {'wfixed': wfixed,
                                                    'wstart': wstart,
                                                    'delta_w': delta_w,
                                                    'instancefile': instancefile,
                                                    'inventoryfile_penalty_threshold': args.inventoryfile_penalty_threshold,
                                                    'penalty_weights_unique': penalty_weights_unique,
                                                    'penalty_weights_conflict': penalty_weights_conflict,
                                                    'store_inventory': True,
                                                    'use_snapshots': True}
                solveAndFindPenaltyThresholdArgs.update(solve_instance_args)
                solveAndFindPenaltyThreshold(**solveAndFindPenaltyThresholdArgs)
if __name__ == "__main__":
    main()
