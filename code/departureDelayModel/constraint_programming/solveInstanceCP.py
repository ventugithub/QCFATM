#!/usr/bin/env python
import numpy as np
import Numberjack as nj
import multiprocessing
import argparse
import os
import sys
import h5py
sys.path.append('../qubo')
try:
    import instance
    import variable
except:
    raise ValueError('Unable to load instance and variable class')


def main():
    parser = argparse.ArgumentParser(description='Solve ATM instance with constraint programming optimization solver',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', required=True, help='input instance file (ignoring the delays included)')
    parser.add_argument('-o', '--output', default='data/results/', help='result folder')
    parser.add_argument('-d', '--maxDelay', default=18, help='Maximum delay', type=int)
    parser.add_argument('-n', '--numDelays', nargs='+', default=[6], help='List of (number of delay steps - 1)', type=int)
    parser.add_argument('--deltat', default=3, help='Temporal threshold for conflicts', type=int)
    parser.add_argument('--np', default=1, help='Number of processes', type=int)
    parser.add_argument('--use_snapshots', action='store_true', help='use snapshot files')
    parser.add_argument('--verbose', action='store_true', help='verbose output')
    parser.add_argument('--timeout', default=None, help='timeout in seconds for exact solver')
    parser.add_argument('--inventory', default='data/inventory.h5', help='Inventory file')
    args = parser.parse_args()
    solve_instances(instancefiles=[args.input],
                    numDelays=args.numDelays,
                    maxDelay=args.maxDelay,
                    deltat=args.deltat,
                    np=args.np,
                    outputFolder=args.output,
                    use_snapshots=args.use_snapshots,
                    verbose=args.verbose,
                    timeout=args.timeout,
                    inventoryfile=args.inventory)

def exists(filename, group):
    """ Check if group or dataset exists HDF5 file """
    if not os.path.exists(filename):
        return False
    f = h5py.File(filename, 'r')
    exists = group in f
    f.close()
    return exists

def solve_instance(instancefile, Nd, maxDelay, deltat, outputFolder, use_snapshots=False, verbose=False, timeout=None, inventoryfile=None):
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    if not os.path.exists(os.path.dirname(inventoryfile)):
        os.makedirs(os.path.dirname(inventoryfile))
    inst = instance.Instance(instancefile)

    # get basename by removing the delayStep and maxDelay values from filename (since they are ignored)
    basename = os.path.basename(instancefile).partition('_delayStep')[0]
    resultfile = "%s/%s.results.h5" % (outputFolder, basename)
    cpsol = 'solution_constraint_programming_numDelays%03i_maxDelay%03i' % (Nd, maxDelay)
    if not exists(resultfile, cpsol) or not use_snapshots:
        Nf = len(inst.flights)
        Nk = len(inst.conflicts)

        # define Nf integer variables from [0, Nd - 1]
        d = nj.VarArray(Nf, Nd)

        # create model
        model = nj.Model()
        model += nj.Minimise(nj.Sum(d))
        for k in range(Nk):
            dtmin = int(inst.timeLimits[k][0])
            dtmax = int(inst.timeLimits[k][0])
            f1 = int(inst.conflicts[k][0])
            f2 = int(inst.conflicts[k][1])
            i = int(inst.flights.index(f1))
            j = int(inst.flights.index(f2))
            constraint1 = maxDelay * (d[i] - d[j]) >= Nd * (deltat - dtmin)
            constraint2 = maxDelay * (d[i] - d[j]) <= - Nd * (deltat + dtmax)
            model += nj.Disjunction([constraint1, constraint2])

        # load solver
        solver = model.load("Mistral")
        if verbose:
            solver.setVerbosity(1)

        # solve
        solver.solve()

        # parse solution
        s = d.solution()
        solution = [int(n) for n in s.strip("[]").split(', ')]

        # check solution
        valids = []
        for k in range(Nk):
            dtmin = int(inst.timeLimits[k][0])
            dtmax = int(inst.timeLimits[k][0])
            f1 = int(inst.conflicts[k][0])
            f2 = int(inst.conflicts[k][1])
            i = int(inst.flights.index(f1))
            j = int(inst.flights.index(f2))
            constraint1 = maxDelay * (solution[i] - solution[j]) >= Nd * (deltat - dtmin)
            constraint2 = maxDelay * (solution[i] - solution[j]) <= - Nd * (deltat + dtmax)
            valid = True if (constraint1 or constraint2) else False
            valids.append(valid)
        if not all(valids):
            print "Solution is not valid. Will not be stored"
        else:
            print "Write valid solution to %s" % resultfile
            delayValues = float(maxDelay) / Nd * np.array(solution, dtype=int)
            delayVariables = variable.IntegerVariable(delayValues)
            delayVariables.save_hdf5(resultfile, name=cpsol)
            # calculate objective function
            totaldelay = 0
            for k in range(Nf):
                totaldelay += delayValues[k]
            f = h5py.File(resultfile, 'a')
            f[cpsol].attrs['total delay'] = totaldelay
            f.close()

def solve_instances(instancefiles, numDelays, np=1, **kwargs):
    if np != 1:
        pool = multiprocessing.Pool(processes=np)
    for instancefile in instancefiles:
        for Nd in numDelays:
            print "Process instance file %s with numDelays=%03i" % (instancefile, Nd)
            solve_instance_args = {'instancefile': instancefile,
                                   'Nd': Nd}
            solve_instance_args.update(kwargs)
            if np != 1:
                pool.apply_async(solve_instance, kwds=solve_instance_args)
            else:
                solve_instance(**solve_instance_args)

    if np != 1:
        pool.close()
        pool.join()

if __name__ == "__main__":
    main()
