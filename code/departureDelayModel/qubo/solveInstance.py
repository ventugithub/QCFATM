#!/usr/bin/env python
import argparse
import os
import subprocess
import multiprocessing
import filelock
import h5py
import numpy as np

import qubo
import solver
import polynomial
import variable

import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Map ATM instance to qubo and solve it')
    parser.add_argument('-i', '--input', required=True, help='input instance yaml file')
    parser.add_argument('-o', '--output', help='result folder')
    parser.add_argument('--num_embed', default=1, help='number of different embeddings', type=int)
    parser.add_argument('--use_snapshots', action='store_true', help='use snapshot files')
    parser.add_argument('--qubo_creation_only', action='store_true', help='qubo creation only')
    parser.add_argument('--embedding_only', action='store_true', help='no quantum annealing')
    parser.add_argument('--retry_embedding', default=0, help='Number of retrys after embedding failed', type=int)
    parser.add_argument('--retry_embedding_desperate', default=0, help='Number of retrys with exteme variable after embedding failed for retry_embedding number of times', type=int)
    parser.add_argument('--unary', action='store_true', help='Use unary representation of integer variables instead of binary representation')
    parser.add_argument('--verbose', action='store_true', help='verbose output')
    parser.add_argument('--store_everything', action='store_true', help='store everything (e.g. physical raw solution)')
    parser.add_argument('--timeout', default=None, help='timeout in seconds for exact solver')
    parser.add_argument('--chimera_m', default=None, help='Number of rows in Chimera', type=int)
    parser.add_argument('--chimera_n', default=None, help='Number of columns in Chimera', type=int)
    parser.add_argument('--chimera_t', default=None, help='Half number of qubits in unit cell of Chimera', type=int)
    parser.add_argument('--exact', action='store_true', help='calculate exact solution with maxsat solver')
    parser.add_argument('--retry_exact', action='store_true', help='retry exact solution in case of previous failure')
    parser.add_argument('--inventory', default='data/inventory.csv', help='Inventory file')
    parser.add_argument('-p2', '--penalty_weight_unique', default=1, help='penaly weight for the term in the QUBO which enforces uniqueness', type=float)
    parser.add_argument('-p3', '--penalty_weight_conflict', default=1, help='penaly weight for the conflict term in the QUBO', type=float)
    args = parser.parse_args()
    chimera = [args.chimera_m, args.chimera_n, args.chimera_t]
    if (any(chimera) and not args.embedding_only):
        parser.error('You can only specify the Chimera parameters if the --embedding_only option is set')
    elif (any(chimera) and not all(chimera)):
        parser.error('You need to specify all 3 Chimera parameters')
    elif (not any(chimera)):
        chimera = {}
    else:
        chimera = {'m': args.chimera_m, 'n': args.chimera_n, 't': args.chimera_t}
    penalty_weights = {
        'unique': args.penalty_weight_unique,
        'conflict': args.penalty_weight_conflict
    }
    solve_instance(instancefile=args.input,
                   outputFolder=args.output,
                   num_embed=args.num_embed,
                   use_snapshots=args.use_snapshots,
                   embedding_only=args.embedding_only,
                   qubo_creation_only=args.qubo_creation_only,
                   retry_embedding=args.retry_embedding,
                   retry_embedding_desperate=args.retry_embedding_desperate,
                   unary=args.unary,
                   verbose=args.verbose,
                   timeout=args.timeout,
                   chimera=chimera,
                   exact=args.exact,
                   store_everything=args.store_everything,
                   penalty_weights=penalty_weights,
                   retry_exact=args.retry_exact,
                   inventoryfile=args.inventory)

def exists(filename, group):
    """ Check if group or dataset exists HDF5 file """
    if not os.path.exists(filename):
        return False
    f = h5py.File(filename, 'r')
    exists = group in f
    f.close()
    return exists

def save_array(array, filename, name, mode='a'):
    """ Save numpy array as dataset in HDF5 file """
    f = h5py.File(filename, mode)
    if name in f:
        del f[name]
    f.create_dataset(name, data=array)
    f.close()

def save_array_in_group(array, filename, groupname, datasetname, mode='a'):
    """ Save numpy array as dataset in HDF5 file """
    f = h5py.File(filename, mode)
    if groupname not in f:
        group = f.create_group(groupname)
    else:
        group = f[groupname]
    if datasetname in group:
        del group[datasetname]
    group.create_dataset(datasetname, data=array)
    f.close()

def solve_instance(instancefile, outputFolder, penalty_weights, num_embed=1, use_snapshots=False, embedding_only=False, qubo_creation_only=False, retry_embedding=0, retry_embedding_desperate=0, unary=False, verbose=False, timeout=None, exact=False, chimera={}, inventoryfile=None, accuracy=14, store_everything=False, retry_exact=False):

    # invertory data
    inventorydata = {}
    if not unary:
        raise ValueError('Binary representation is not feasible for this model due to the conflict penalizing term in the cost function')
    # string representing the penalty weights
    pwstr = "pw"
    for k, v in penalty_weights.items():
        pwstr = pwstr + "-%s%0.3f" % (k, v)
    # create folders if necessary
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    if not os.path.exists(os.path.dirname(inventoryfile)):
        os.makedirs(os.path.dirname(inventoryfile))
    resultfile = "%s/%s.results.h5" % (outputFolder, os.path.basename(instancefile).rstrip('.h5'))
    hardConstraints = ['conflict', 'unique']

    # read in instance and calculate QUBO and index mapping
    subqubonames = ['departure', 'conflict', 'unique']
    grouplist = []
    grouplist.append('%s/qubo' % pwstr)
    for name in subqubonames:
        grouplist.append('%s/subqubo-%s' % (pwstr, name))
    grouplist.append('variable')
    if not all([exists(resultfile, g) for g in grouplist]) or not use_snapshots:
        print "Calculate QUBO ..."
        q, subqubos, var = qubo.get_qubo(instancefile, penalty_weights, unary)
        var.save_hdf5(resultfile)
        q.save_hdf5(resultfile, '%s/qubo' % pwstr)
        for name in subqubonames:
            subqubos[name].save_hdf5(resultfile, '%s/subqubo-%s' % (pwstr, name))
    else:
        print "Read in QUBO ..."
        q = polynomial.Polynomial()
        q.load_hdf5(resultfile, '%s/qubo' % pwstr)
        subqubos = {}
        for name in subqubonames:
            subqubos[name] = polynomial.Polynomial()
            subqubos[name].load_hdf5(resultfile, '%s/subqubo-%s' % (pwstr, name))
        print "Read in Variable ..."
        if unary:
            var = variable.Unary(resultfile, instancefile, hdf5=True)

    print "Coefficient range ratio of QUBO: (maxLinear/minLinear, maxQuadratic/minQuadratic) = ", q.getCoefficientRange()
    inventorydata['maxCoefficientRangeRatio'] = max(q.getCoefficientRange())
    inventorydata['coefficientRangeRatio'] = {}
    for k in subqubonames:
        inventorydata['coefficientRangeRatio'][k] = subqubos[k].getCoefficientRange()
        print "Coefficient range ratio of Sub-QUBO %s: (maxLinear/minLinear, maxQuadratic/minQuadratic) = " % k, inventorydata['coefficientRangeRatio'][k]
    N = 0
    for k in q.poly.keys():
        if len(k) == 1:
            N = N + 1
    print "Number of logical Qubits: %i" % N
    inventorydata['NLogQubits'] = N

    if qubo_creation_only:
        return

    ###################################
    # solve problem exactly
    ###################################
    s = solver.Solver(q)
    energyExact = None
    exactSuccess = False
    inventorydata['exactValid'] = np.nan
    inventorydata['exactEnergy'] = np.nan
    if exact:
        exactSolutionStr = '%s/exactSolution' % pwstr
        datasets = []
        datasets.append('%s/rawExactSolution' % exactSolutionStr)
        datasets.append('%s/energy' % exactSolutionStr)
        if not exists(resultfile, exactSolutionStr) or not use_snapshots:
            print "Calculate exact solution ..."
            rawresult = s.solve_exact(timeout=timeout)
            if not rawresult:
                print "No exact solution found. Timeout was ", timeout, "seconds"
                f = h5py.File(resultfile, 'a')
                f[exactSolutionStr].attrs['foundSolution'] = False
                f.close()
            else:
                exactSuccess = True
                f = h5py.File(resultfile, 'a')
                if exactSolutionStr not in f:
                    f.create_group(exactSolutionStr)
                f[exactSolutionStr].attrs['foundSolution'] = True
                f.close()
            if exactSuccess:
                save_array_in_group(rawresult['solution'], resultfile, exactSolutionStr, 'rawExactSolution')
                save_array_in_group(rawresult['energy'], resultfile, exactSolutionStr, 'energy')
        else:
            print "Read in exact solution ..."
            f = h5py.File(resultfile, 'r')
            if exactSolutionStr in f and 'foundSolution' in f[exactSolutionStr].attrs and f[exactSolutionStr].attrs['foundSolution']:
                exactSuccess = True
                rawresult = {}
                rawresult['solution'] = f['%s/rawExactSolution' % exactSolutionStr].value.tolist()
                rawresult['energy'] = f['%s/energy' % exactSolutionStr].value
                f.close()
            elif retry_exact:
                f.close()
                print "Calculate exact solution ..."
                rawresult = s.solve_exact(timeout=timeout)
                if not rawresult:
                    print "No exact solution found. Timeout was ", timeout, "seconds"
                    f = h5py.File(resultfile, 'a')
                    f[exactSolutionStr].attrs['foundSolution'] = False
                    f.close()
                else:
                    exactSuccess = True
                    f = h5py.File(resultfile, 'a')
                    f[exactSolutionStr].attrs['foundSolution'] = True
                    f.close()
            else:
                print "Warning: No exact solution available in file. Probably due to timeout. Use --retry_exact for recalculation"
                f.close()
        if exactSuccess:
            energyExact = rawresult['energy']
            inventorydata['exactEnergy'] = energyExact
            print "Exact solution has energy: %f" % energyExact
            for k, v in subqubos.items():
                print "Contribution of %s term: %f" % (k, v.evaluate(rawresult['solution']))

            if any([subqubos[k].evaluate(rawresult['solution']) for k in hardConstraints]):
                f = h5py.File(resultfile, 'a')
                f[exactSolutionStr].attrs['isValid'] = False
                f.close()
                print "Exact solution is NOT VALID"
                inventorydata['exactValid'] = False
            else:
                f = h5py.File(resultfile, 'a')
                f[exactSolutionStr].attrs['isValid'] = True
                f.close()
                print "Exact solution is VALID"
                inventorydata['exactValid'] = True

            ###################################
            # map solution vector back to
            # multi-indices
            ###################################
            if not exists(resultfile, '%s/exactSolution' % exactSolutionStr) or not use_snapshots:
                print "Map exact solution to integers ..."
                result = var.getIntegerVariables(rawresult['solution'])
                print "Write exact solution ..."
                result.save_hdf5(resultfile, name='%s/exactSolution' % exactSolutionStr)
            else:
                print "Read in exact solution ..."
                result = variable.IntegerVariable(resultfile, '%s/exactSolution' % exactSolutionStr, hdf5=True)
    ###################################
    # get embedding
    ###################################
    # get number of physical Qubits available
    qubits = []
    hwa = s.getHardwareAdjacency(use_snapshots=True)
    if (chimera):
        hwa = solver.get_chimera_adjacency(**chimera)
    for u, v in hwa:
        qubits.append(u)
        qubits.append(v)
    Nmax = len(list(set(qubits)))
    if N > Nmax:
        print "Number of logical qubits exceeds number of physical qubits. No embedding possible"
        exit(0)

    # get embedding
    inventorydata['embedding'] = {}
    embedparams = {}
    skipEmbedding = False
    for e in range(num_embed):
        # default values for unsuccessfull runs
        inventorydata['embedding'][e] = {}
        inventorydata['embedding'][e]['maxCoefficientRangeRatio'] = np.nan
        inventorydata['embedding'][e]['successProbability'] = np.nan
        inventorydata['embedding'][e]['repeatTo99'] = np.nan
        inventorydata['embedding'][e]['valid'] = np.nan
        inventorydata['embedding'][e]['energy'] = np.nan
        print "Embedding %i" % e
        embedname = "%s/embedding%05i" % (pwstr, e)
        if (chimera):
            embedname = "%s/embedding%05i_chimera%03i_%03i_%03i" % (pwstr, e, chimera['m'], chimera['n'], chimera['t'])
        if not exists(resultfile, embedname) or not use_snapshots:
            if not skipEmbedding:
                print "Calculate embedding ..."
                s.calculateEmbedding(eIndex=e, verbose=verbose, **embedparams)
                s.writeEmbeddingHDF5(resultfile, embedname, eIndex=e)
            else:
                print "Skip calculation of embedding %i ..." % e
                s.embeddings[e] = []
                s.writeEmbeddingHDF5(resultfile, embedname, eIndex=e)
        else:
            print "Read in embedding ..."
            s.readEmbeddingHDF5(resultfile, embedname, eIndex=e)
        NPhysQubits = len([item for sublist in s.embeddings[e] for item in sublist])
        inventorydata['embedding'][e]['NPhysQubits'] = NPhysQubits
        print "Number of physical Qubits (0: embedding unsuccessful): %i" % NPhysQubits
        if not any(s.embeddings.values()) and e >= retry_embedding:
            if e < retry_embedding + retry_embedding_desperate:
                print "Set embedding parameters to extreme values for the following"
                embedparams['max_no_improvement'] = 30
                embedparams['timeout'] = 500
                embedparams['tries'] = 50
            else:
                print "Last %i embeddings were unsuccessfull. Skip the following." % (retry_embedding + retry_embedding_desperate + 1)
                skipEmbedding = True

        if not embedding_only and NPhysQubits:
            ###################################
            # solve problem
            ###################################
            num_reads = 10000
            solutionStr = "%s/solution" % pwstr
            datasets = []
            datasets.append('%s/quboEmbedded' % solutionStr)
            datasets.append('%s/logRawSolutions' % solutionStr)
            datasets.append('%s/energies' % solutionStr)
            datasets.append('%s/numOccurences' % solutionStr)
            if not all([exists(resultfile, d) for d in datasets]) or (not exists(resultfile, '%s/physRawSolutions' % solutionStr) and store_everything) or not use_snapshots:
                print "Calculate solutions ..."
                physRawResult, logRawResult, energies, numOccurrences = s.solve(num_reads=num_reads, eIndex=e)

                if (verbose):
                    r = s.solve_embedded_exact(e, timeout=timeout)
                    if energies[0] == r['energy']:
                        print "Annealer found the correct solution"
                    else:
                        print "Warning: Annealer did not find the correct solution"
                qubo_embedded = s.getEmbeddedQUBO(e, suppressThreshold=1E-14)

                if store_everything:
                    save_array_in_group(physRawResult, resultfile, solutionStr, 'physRawSolutions')
                save_array_in_group(logRawResult, resultfile, solutionStr, 'logRawSolutions')
                save_array_in_group(energies, resultfile, solutionStr, 'energies')
                save_array_in_group(numOccurrences, resultfile, solutionStr, 'numOccurences')
                qubo_embedded.save_hdf5(resultfile, '%s/quboEmbedded' % solutionStr)
            else:
                print "Read in solution ..."
                qubo_embedded = polynomial.Polynomial()
                qubo_embedded.load_hdf5(resultfile, '%s/quboEmbedded' % solutionStr)
                f = h5py.File(resultfile, 'r')
                if store_everything:
                    physRawResult = f['%s/physRawSolutions' % solutionStr].value
                logRawResult = f['%s/logRawSolutions' % solutionStr].value
                energies = f['%s/energies' % solutionStr].value
                numOccurrences = f['%s/numOccurences' % solutionStr].value
                f.close()
            crr = qubo_embedded.getCoefficientRange()
            print "Coefficient range ratio of embedded QUBO: (maxLinear/minLinear, maxQuadratic/minQuadratic) = ", crr
            inventorydata['embedding'][e]['maxCoefficientRangeRatio'] = max(crr[0], crr[1])
            energy = q.evaluate(logRawResult[0])
            inventorydata['embedding'][e]['energy'] = energy
            print "Solution has energy: %f" % energy
            for k, v in subqubos.items():
                print "Contribution of %s term: %f" % (k, v.evaluate(logRawResult[0]))

            # analyse results
            if exact:
                print "Analyse Annealing runs ..."
                # calculate the energies of the solutions from the qubo
                energiesFromQUBO = [q.evaluate(sol) for sol in logRawResult]
                # create dataframe with columns: energies, number of occurences
                data = pd.DataFrame({'Energies': np.array(energiesFromQUBO), 'NumOcc': numOccurrences}).round(accuracy)
                # sort by energies
                data.sort_values(by='Energies', inplace=True)
                # get success probability
                numOccPerEnergy = pd.groupby(data, 'Energies').sum()
                energyExactRounded = np.round(energyExact, accuracy)
                if energyExactRounded in numOccPerEnergy.index:
                    NSuccess = numOccPerEnergy.loc[np.round(energyExact, accuracy)][0]
                    NRuns = numOccPerEnergy['NumOcc'].sum()
                    successProbability = NSuccess/float(NRuns)
                    repeatTo99 = np.log(1-0.99)/np.log(1 - successProbability)
                else:
                    successProbability = 0
                    repeatTo99 = np.inf

                print "Success Probability is ", successProbability
                print "Number of runs until 99% success is ", repeatTo99
                inventorydata['embedding'][e]['successProbability'] = successProbability
                inventorydata['embedding'][e]['repeatTo99'] = repeatTo99

            if any([subqubos[k].evaluate(logRawResult[0]) for k in hardConstraints]):
                f = h5py.File(resultfile, 'a')
                f[solutionStr].attrs['isValid'] = False
                f.close()
                inventorydata['embedding'][e]['valid'] = False
                print "Solution is NOT VALID"
            else:
                f = h5py.File(resultfile, 'a')
                f[solutionStr].attrs['isValid'] = False
                f.close()
                inventorydata['embedding'][e]['valid'] = True
                print "Solution is VALID"

            ###################################
            # map solution vector back to
            # multi-indices
            ###################################
            if not exists(resultfile, '%s/solution' % solutionStr) or not use_snapshots:
                print "Map solution to integers ..."
                result = var.getIntegerVariables(logRawResult[0])
                print "Write solution ..."
                result.save_hdf5(resultfile, '%s/solution' % solutionStr)
            else:
                print "Read in solution ..."
                result = variable.IntegerVariable(resultfile, '%s/solution' % solutionStr, hdf5=True)

    # add data to inventory
    if inventoryfile:
        repoversion = subprocess.check_output(['git', 'rev-parse', 'HEAD']).rstrip('\n')
        embeddings = inventorydata['embedding'].keys()
        NRows = len(embeddings) + 1
        inventory = pd.DataFrame({'instance': [instancefile] * NRows,
                                  'exact': np.append(np.array([True]), np.array([False] * (NRows - 1))),
                                  'embedding': np.append(np.array([np.nan]), np.array(embeddings, dtype=int)),
                                  'penalty_weight_unique': [penalty_weights['unique']] * NRows,
                                  'penalty_weight_conflict': [penalty_weights['conflict']] * NRows,
                                  'NLogQubits': np.array(inventorydata['NLogQubits']),
                                  'NPhysQubits': np.append(np.array([np.nan]), np.array([inventorydata['embedding'][e]['NPhysQubits'] for e in embeddings])),
                                  'SuccessProbability': np.round(np.append(np.array([np.nan]), np.array([inventorydata['embedding'][e]['successProbability'] for e in embeddings])), 5),
                                  'repeatTo99': np.round(np.append(np.array([np.nan]), np.array([inventorydata['embedding'][e]['repeatTo99'] for e in embeddings])), 5),
                                  'isValid': np.append(np.array([inventorydata['exactValid']]), np.array([inventorydata['embedding'][e]['valid'] for e in embeddings])),
                                  'energy': np.append(np.array([inventorydata['exactEnergy']]), np.array([inventorydata['embedding'][e]['energy'] for e in embeddings])),
                                  'maxCoefficientRangeRatioEmbedded': np.append(np.array([np.nan]), np.array([inventorydata['embedding'][e]['maxCoefficientRangeRatio'] for e in embeddings])),
                                  'maxCoefficientRangeRatio': [inventorydata['maxCoefficientRangeRatio']] * NRows,
                                  'version': [repoversion] * NRows
                                  })
        inventory.set_index('instance', inplace=True)

        # use lock file to prevent simultaneous updates to inventory
        # in the case of multiprocessing
        lockfile = inventoryfile + ".lock"
        lock = filelock.FileLock(lockfile)
        with lock.acquire():
            # read in inventory file if existent
            if os.path.exists(inventoryfile):
                inventory_before = pd.read_hdf(inventoryfile, 'inventory')
                inventory = pd.concat([inventory_before, inventory])

            # drop duplicates but ignore version
            columnsToConsider = inventory.columns.values.tolist()
            columnsToConsider.remove('version')
            inventory.drop_duplicates(subset=columnsToConsider, keep='last', inplace=True)
            inventory.to_hdf(inventoryfile, 'inventory', mode='w')

def solve_instances(instancefiles, penalty_weights, np=1, **kwargs):
    if np != 1:
        pool = multiprocessing.Pool(processes=np)
    for instancefile in instancefiles:
        print "Process instance file %s" % instancefile
        solve_instance_args = {'instancefile': instancefile, 'penalty_weights': penalty_weights}
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
