#!/usr/bin/env python
import numpy as np
import Numberjack as nj
import multiprocessing
import filelock
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
    parser.add_argument('-n', '--numDelays', nargs='+', default=[6], help='List of (number of delay steps - 1), Set to 0 for continuous variables', type=int)
    parser.add_argument('--deltat', default=3, help='Temporal threshold for conflicts', type=int)
    parser.add_argument('--np', default=1, help='Number of processes', type=int)
    parser.add_argument('--use_snapshots', action='store_true', help='use snapshot files')
    parser.add_argument('--skipBigProblems', help='Do not try to solve instance bigger than this number', type=int)
    parser.add_argument('--verbose', action='store_true', help='verbose output')
    parser.add_argument('--timeout', default=None, help='timeout in seconds for exact solver', type=int)
    parser.add_argument('--inventory', default='data/inventory.h5', help='Inventory file')
    parser.add_argument('--accuracy', default=0.0, help='Accuracy to fulfill constraints in continous case (make up for rounding errors', type=float)
    args = parser.parse_args()
    solve_instances(instancefiles=[args.input],
                    numDelays=args.numDelays,
                    maxDelay=args.maxDelay,
                    deltat=args.deltat,
                    np=args.np,
                    outputFolder=args.output,
                    use_snapshots=args.use_snapshots,
                    skipBigProblems=args.skipBigProblems,
                    accuracy=args.accuracy,
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

def solve_instance(instancefile, Nd, maxDelay, outputFolder, deltat=3, skipBigProblems=None, use_snapshots=False, accuracy=0.0, verbose=False, timeout=None, inventoryfile=None):
    print "Process instance file %s with numDelays=%03i, maxDelay=%03i" % (instancefile, Nd, maxDelay)
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    if not os.path.exists(os.path.dirname(inventoryfile)):
        os.makedirs(os.path.dirname(inventoryfile))
    inst = instance.Instance(instancefile)

    # if argument Nd is zero, assume continous delay variable
    discrete = True
    if Nd == 0:
        discrete = False
        deltat = float(deltat)

    # get basename by removing the delayStep and maxDelay values from filename (since they are ignored)
    basename = os.path.basename(instancefile).partition('_delayStep')[0]
    resultfile = "%s/%s.results.h5" % (outputFolder, basename)
    cpsol = 'solution_constraint_programming_numDelays%03i_maxDelay%03i' % (Nd, maxDelay)
    if not exists(resultfile, cpsol) or not use_snapshots:
        Nf = len(inst.flights)
        Nk = len(inst.conflicts)

        # define Nf integer variables from [0, Nd]
        if discrete:
            d = nj.VarArray(Nf, Nd + 1)
        # define Nf float variables from [0, Nd] if argument Nd == 0
        else:
            d = nj.VarArray(Nf, 0.0, float(maxDelay))

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
            if discrete:
                constraint1 = maxDelay * (d[i] - d[j]) >= Nd * (deltat - dtmin)
                constraint2 = maxDelay * (d[i] - d[j]) <= - Nd * (deltat + dtmax)
            else:
                constraint1 = (d[i] - d[j]) >= (deltat - dtmin)
                constraint2 = (d[i] - d[j]) <= - (deltat + dtmax)
            model += nj.Disjunction([constraint1, constraint2])

        # load solver
        if discrete:
            solver = model.load("Mistral")
        else:
            solver = model.load("CPLEX")
        if verbose:
            solver.setVerbosity(1)
        if timeout:
            solver.setTimeLimit(timeout)

        # solve
        skipProblem = False
        if skipBigProblems is None:
            success = solver.solve()
        else:
            problemSize = len(model.variables) + len(model.constraints)
            if problemSize <= skipBigProblems:
                success = solver.solve()
            else:
                success = False
                skipProblem = True
        if success:

            # parse solution
            s = d.solution()
            if discrete:
                solution = [int(n) for n in s.strip("[]").split(', ')]
            else:
                solution = [float(n) for n in s.strip("[]").split(', ')]

            # check solution
            valids = []
            for k in range(Nk):
                dtmin = int(inst.timeLimits[k][0])
                dtmax = int(inst.timeLimits[k][0])
                f1 = int(inst.conflicts[k][0])
                f2 = int(inst.conflicts[k][1])
                i = int(inst.flights.index(f1))
                j = int(inst.flights.index(f2))
                if discrete:
                    constraint1 = maxDelay * (solution[i] - solution[j]) >= Nd * (deltat - dtmin)
                    constraint2 = maxDelay * (solution[i] - solution[j]) <= - Nd * (deltat + dtmax)
                else:
                    constraint1 = (solution[i] - solution[j]) >= (deltat - dtmin) - accuracy
                    constraint2 = (solution[i] - solution[j]) <= - (deltat + dtmax) + accuracy

                valid = True if (constraint1 or constraint2) else False
                valids.append(valid)
            if not all(valids):
                print "Solution is not valid. Will not be stored"
                if verbose:
                    for k in range(Nk):
                        dtmin = int(inst.timeLimits[k][0])
                        dtmax = int(inst.timeLimits[k][0])
                        f1 = int(inst.conflicts[k][0])
                        f2 = int(inst.conflicts[k][1])
                        i = int(inst.flights.index(f1))
                        j = int(inst.flights.index(f2))
                        if discrete:
                            diff = maxDelay / Nd * (solution[i] - solution[j])
                        else:
                            diff = solution[i] - solution[j]
                        print "Constraint violation in solution:", deltat - dtmin, "not <=", diff, " or ", diff, "not >=", -deltat - dtmax, "( dtmin =", dtmin, "dtmax =", dtmax, ")"

            else:
                if discrete:
                    delayValues = float(maxDelay) / Nd * np.array(solution, dtype=int)
                else:
                    delayValues = solution
                delayVariables = variable.IntegerVariable(delayValues)
                # calculate objective function
                totaldelay = 0
                for k in range(Nf):
                    totaldelay += delayValues[k]
                print "Write valid solution to %s" % resultfile
                # use lock file to prevent simultaneous updates to inventory
                # in the case of multiprocessing
                lockfile = resultfile + ".lock"
                lock = filelock.FileLock(lockfile)
                with lock.acquire():
                    delayVariables.save_hdf5(resultfile, name=cpsol)
                    f = h5py.File(resultfile, 'a')
                    f[cpsol].attrs['total delay'] = totaldelay
                    f.close()
        else:
            if skipProblem:
                print "Skip big problem (%i > %i)" % (problemSize, skipBigProblems)
            elif solver.is_unsat():
                print "No solution found. Problem not satisfiable"
            else:
                print "No solution found. Nothing will be stored. Runtime was %s. Timeout was %s" % (solver.getTime(), timeout)

def solve_instances(instancefiles, outputFolders, maxDelays, numDelays, np=1, **kwargs):
    """
    Runs the CP solver in parallel for various instances and parameters

    instancefiles: list of instance files to process
    outputFolder: dictonary with key: instancefile, value: outputFolder
    maxDelays: dictionary with key: instancefile, value: list of maxDelays
    numDelays: dictionary with key: tuple: (instancefile, maxDelay), value: list of numDelays
    """
    # check:
    print "Check Arguments of parallel solver ..."
    if set(instancefiles) != set(outputFolders.keys()):
        raise ValueError('Number of instancefiles is not equal to the number of outputFolders')
    if set(maxDelays.keys()) != set(instancefiles):
        raise ValueError('Not all instancefiles are in the keys of maxDelays')
    for k, v in numDelays.items():
        if not k[0] in instancefiles:
            raise ValueError('Not all instancefiles are in the keys of numDelays')
        if not k[1] in maxDelays[k[0]]:
            raise ValueError('Not all maxDelays are in the keys of numDelays')
        for numDelay in v:
            if numDelay > maxDelays[k[0]]:
                raise ValueError('numDelay value is greater than maxDelay')

    print "Run %s instances on %i cores ..." % (len(instancefiles), np)
    if np != 1:
        pool = multiprocessing.Pool(processes=np)
    for instancefile in instancefiles:
        outputFolder = outputFolders[instancefile]
        maxDelayList = maxDelays[instancefile]
        for maxDelay in maxDelayList:
            numDelayList = numDelays[(instancefile, maxDelay)]
            for Nd in numDelayList:
                solve_instance_args = {'instancefile': instancefile,
                                       'outputFolder': outputFolder,
                                       'maxDelay': maxDelay,
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
