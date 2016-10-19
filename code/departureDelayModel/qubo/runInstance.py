#!/usr/bin/env python
import argparse
import os
import yaml
import subprocess
import numpy as np

import qubo
import solver
import polynomial
import variable

import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Map ATM instance to qubo and solve it')
    parser.add_argument('-i', '--input', required=True, help='input instance yaml file')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--num_embed', default=1, help='number of different embeddings', type=int)
    group.add_argument('-e', default=0, help='choose only a single embedding by index', type=int)
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
    atm(instancefile=args.input,
        num_embed=args.num_embed,
        e=args.e,
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

def atm(instancefile, penalty_weights, num_embed=1, e=None, use_snapshots=False, embedding_only=False, qubo_creation_only=False, retry_embedding=0, retry_embedding_desperate=0, unary=False, verbose=False, timeout=None, exact=False, chimera={}, inventoryfile=None, accuracy=14, store_everything=False, retry_exact=False):

    # invertory data
    inventorydata = {}
    if not unary:
        raise ValueError('Binary representation is not feasible for this model due to the conflict penalizing term in the cost function')
    representation = 'binary'
    if unary:
        representation = 'unary'
    # add penalty weights to representation string in order
    # to distinguish between different penalty weights
    representation = representation + ".pw"
    for k, v in penalty_weights.items():
        representation = representation + "-%s%0.3f" % (k, v)
    # read in instance and calculate QUBO and index mapping
    qubofile = "%s.%s.qubo.yaml" % (instancefile, representation)
    subqubofiles = {}
    subqubofiles['departure'] = "%s.%s.subqubo-departure.yaml" % (instancefile, representation)
    subqubofiles['conflict'] = "%s.%s.subqubo-conflict.yaml" % (instancefile, representation)
    subqubofiles['unique'] = "%s.%s.subqubo-unique.yaml" % (instancefile, representation)
    variablefile = "%s.%s.variable.yaml" % (instancefile, representation)
    hardConstraints = ['conflict', 'unique']

    if not os.path.exists(qubofile) or not any([os.path.exists(f) for f in subqubofiles.values()]) or not os.path.exists(variablefile) or not use_snapshots:
        print "Calculate QUBO ..."
        q, subqubos, var = qubo.get_qubo(instancefile, penalty_weights, unary)
        var.save(variablefile)
        q.save(qubofile)
        for k in subqubofiles.keys():
            subqubos[k].save(subqubofiles[k])
    else:
        print "Read in QUBO ..."
        q = polynomial.Polynomial()
        q.load(qubofile)
        subqubos = {}
        for k in subqubofiles.keys():
            subqubos[k] = polynomial.Polynomial()
            subqubos[k].load(subqubofiles[k])
        print "Read in Variable ..."
        if unary:
            var = variable.Unary(variablefile, instancefile)

    print "Coefficient range ratio of QUBO: (maxLinear/minLinear, maxQuadratic/minQuadratic) = ", q.getCoefficientRange()
    inventorydata['maxCoefficientRangeRatio'] = max(q.getCoefficientRange())
    inventorydata['coefficientRangeRatio'] = {}
    for k in subqubofiles.keys():
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
    name = "%s.%s" % (instancefile, representation)
    rawExactSolutionFile = "%s.rawExactSolution.yaml" % name
    isValidFile = "%s.exactSolutionIsValid.txt" % name
    inventorydata['exactValid'] = np.nan
    if exact:
        if not os.path.exists(rawExactSolutionFile) or not use_snapshots:
            print "Calculate exact solution ..."
            rawresult = s.solve_exact(timeout=timeout)
            if not rawresult:
                print "No exact solution found. Timeout was ", timeout, "seconds"
            else:
                exactSuccess = True
            f = open(rawExactSolutionFile, 'w')
            yaml.dump(rawresult, f)
            f.close()
        else:
            print "Read in exact solution ..."
            f = open(rawExactSolutionFile, 'r')
            rawresult = yaml.load(f)
            f.close()
            if rawresult:
                exactSuccess = True
            elif retry_exact:
                print "Calculate exact solution ..."
                rawresult = s.solve_exact(timeout=timeout)
                if not rawresult:
                    print "No exact solution found. Timeout was ", timeout, "seconds"
                else:
                    exactSuccess = True
                f = open(rawExactSolutionFile, 'w')
                yaml.dump(rawresult, f)
                f.close()
            else:
                print "Warning: No exact solution available in file. Probably due to timeout. Use --retry_exact for recalculation"
        if exactSuccess:
            energyExact = rawresult['energy']
            print "Exact solution has energy: %f" % energyExact
            for k, v in subqubos.items():
                print "Contribution of %s term: %f" % (k, v.evaluate(rawresult['solution']))

            if any([subqubos[k].evaluate(rawresult['solution']) for k in hardConstraints]):
                f = open(isValidFile, 'w')
                f.write('not valid\n')
                f.close()
                print "Exact solution is NOT VALID"
                inventorydata['exactValid'] = False
            else:
                f = open(isValidFile, 'w')
                f.write('valid\n')
                f.close()
                print "Exact solution is VALID"
                inventorydata['exactValid'] = True

            ###################################
            # map solution vector back to
            # multi-indices
            ###################################
            exactSolutionFile = "%s.exactSolution.yaml" % name
            if not os.path.exists(exactSolutionFile) or not use_snapshots:
                print "Map exact solution to integers ..."
                result = var.getIntegerVariables(rawresult['solution'])
                print "Write exact solution to %s ..." % exactSolutionFile
                result.save(exactSolutionFile)
            else:
                print "Read in exact solution ..."
                result = variable.IntegerVariable(exactSolutionFile)

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
        print "Embedding %i" % e
        name = "%s.%s.embedding%05i" % (instancefile, representation, e)
        if (chimera):
            name = "%s.%s.embedding%05i_chimera%03i_%03i_%03i" % (instancefile, representation, e, chimera['m'], chimera['n'], chimera['t'])
        embedfile = "%s.yaml" % name
        if not os.path.exists(embedfile) or not use_snapshots:
            if not skipEmbedding:
                print "Calculate embedding ..."
                s.calculateEmbedding(eIndex=e, verbose=verbose, **embedparams)
                s.writeEmbedding(embedfile, eIndex=e)
            else:
                print "Skip calculation of embedding %i ..." % e
                s.embeddings[e] = []
                s.writeEmbedding(embedfile, eIndex=e)
        else:
            print "Read in embedding ..."
            s.readEmbedding(embedfile, eIndex=e)
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
            quboEmbeddedFile = "%s.quboEmbedded.yaml" % name
            isingEmbeddedFile = "%s.isingEmbedded.txt" % name
            physRawSolutionFile = "%s.physRawSolutions.npy" % name
            logRawSolutionFile = "%s.logRawSolutions.npy" % name
            energiesFile = "%s.energies.npy" % name
            numOccurrencesFile = "%s.numOccurrences.npy" % name
            if (not os.path.exists(physRawSolutionFile) and store_everything) or not os.path.exists(logRawSolutionFile) or not os.path.exists(energiesFile) or not os.path.exists(numOccurrencesFile) and not os.path.exists(quboEmbeddedFile) or not use_snapshots:
                print "Calculate solutions ..."
                physRawResult, logRawResult, energies, numOccurrences = s.solve(num_reads=num_reads, eIndex=e)

                if (verbose):
                    r = s.solve_embedded_exact(e, timeout=timeout)
                    if energies[0] == r['energy']:
                        print "Annealer found the correct solution"
                    else:
                        print "Warning: Annealer did not find the correct solution"
                qubo_embedded = s.getEmbeddedQUBO(e, suppressThreshold=1E-14)
                qubo_embedded.save(quboEmbeddedFile)
                s.saveEmbeddedIsing(isingEmbeddedFile, e)

                if store_everything:
                    np.save(physRawSolutionFile, physRawResult)
                np.save(logRawSolutionFile, logRawResult)
                np.save(energiesFile, energies)
                np.save(numOccurrencesFile, numOccurrences)
            else:
                print "Read in solution ..."
                qubo_embedded = polynomial.Polynomial()
                qubo_embedded.load(quboEmbeddedFile)
                if store_everything:
                    physRawResult = np.load(physRawSolutionFile)
                logRawResult = np.load(logRawSolutionFile)
                energies = np.load(energiesFile)
                numOccurrences = np.load(numOccurrencesFile)
            crr = qubo_embedded.getCoefficientRange()
            print "Coefficient range ratio of embedded QUBO: (maxLinear/minLinear, maxQuadratic/minQuadratic) = ", crr
            inventorydata['embedding'][e]['maxCoefficientRangeRatio'] = max(crr[0], crr[1])
            print "Solution has energy: %f" % q.evaluate(logRawResult[0])
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

            isValidFile = "%s.solutionIsValid.txt" % name
            if any([subqubos[k].evaluate(logRawResult[0]) for k in hardConstraints]):
                f = open(isValidFile, 'w')
                f.write('not valid\n')
                f.close()
                inventorydata['embedding'][e]['valid'] = False
                print "Solution is NOT VALID"
            else:
                f = open(isValidFile, 'w')
                f.write('valid\n')
                f.close()
                inventorydata['embedding'][e]['valid'] = True
                print "Solution is VALID"

            ###################################
            # map solution vector back to
            # multi-indices
            ###################################
            solutionfile = "%s.solution.yaml" % name
            if not os.path.exists(solutionfile) or not use_snapshots:
                print "Map solution to integers ..."
                result = var.getIntegerVariables(logRawResult[0])
                print "Write solution to %s ..." % solutionfile
                result.save(solutionfile)
            else:
                print "Read in solution ..."
                result = variable.IntegerVariable(solutionfile)

    # add data to inventory
    if inventoryfile:
        repoversion = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
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
                                  'maxCoefficientRangeRatioEmbedded': np.append(np.array([np.nan]), np.array([inventorydata['embedding'][e]['maxCoefficientRangeRatio'] for e in embeddings])),
                                  'maxCoefficientRangeRatio': [inventorydata['maxCoefficientRangeRatio']] * NRows,
                                  'version': [repoversion] * NRows
                                  })
        inventory.set_index('instance', inplace=True)

        # read in inventory file if existent
        if os.path.exists(inventoryfile):
            inventory_before = pd.read_csv(inventoryfile, index_col='instance')
            inventory = pd.concat([inventory_before, inventory])

        inventory.reset_index(level=0, inplace=True)
        # drop duplicates but ignore version
        inventory.drop_duplicates(inplace=True, subset=list(inventory.columns).remove('version'))
        inventory.set_index('instance', inplace=True)
        inventory.to_csv(inventoryfile, mode='w')

if __name__ == "__main__":
    main()
