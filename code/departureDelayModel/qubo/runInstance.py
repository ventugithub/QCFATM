#!/usr/bin/env python
import argparse
import os
import yaml
import numpy as np

import qubo
import solver
import polynomial
import variable

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
    parser.add_argument('--retry_embedding_desperate', action='store_true', help='try extreme values for embedding')
    parser.add_argument('--unary', action='store_true', help='Use unary representation of integer variables instead of binary representation')
    parser.add_argument('--verbose', action='store_true', help='verbose output')
    parser.add_argument('--timeout', default=None, help='timeout in seconds for exact solver')
    parser.add_argument('--chimera_m', default=None, help='Number of rows in Chimera', type=int)
    parser.add_argument('--chimera_n', default=None, help='Number of columns in Chimera', type=int)
    parser.add_argument('--chimera_t', default=None, help='Half number of qubits in unit cell of Chimera', type=int)
    parser.add_argument('--exact', action='store_true', help='calculate exact solution with maxsat solver')
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
        exact=args.exact)

def atm(instancefile, num_embed=1, e=None, use_snapshots=False, embedding_only=False, qubo_creation_only=False, retry_embedding=0, retry_embedding_desperate=False, unary=False, verbose=False, timeout=None, exact=False, chimera={}):
    # read in instance and calculate QUBO and index mapping
    qubofile = "%s.qubo.yaml" % instancefile
    subqubofiles = {}
    subqubofiles['departure'] = "%s.subqubo-departure.yaml" % instancefile
    subqubofiles['conflict'] = "%s.subqubo-conflict.yaml" % instancefile
    subqubofiles['boundary-condition'] = "%s.subqubo-boundary-condition.yaml" % instancefile
    subqubofiles['departure-unique'] = "%s.subqubo-departure-unique.yaml" % instancefile
    subqubofiles['conflict-unique'] = "%s.subqubo-conflict-unique.yaml" % instancefile
    variablefile = "%s.variable.yaml" % instancefile
    if not os.path.exists(qubofile) or not any([os.path.exists(f) for f in subqubofiles.values()]) or not os.path.exists(variablefile) or not use_snapshots:
        print "Calculate QUBO ..."
        q, subqubos, var = qubo.get_qubo(instancefile, unary)
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
        else:
            var = variable.Binary(variablefile, instancefile)

    N = 0
    for k in q.poly.keys():
        if len(k) == 1:
            N = N + 1
    print "Number of logical Qubits: %i" % N

    if qubo_creation_only:
        return
    ###################################
    # get embedding
    ###################################
    s = solver.Solver(q)
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
    embedparams = {}
    for e in range(num_embed):
        print "Embedding %i" % e
        name = "%s.embedding%05i" % (instancefile, e)
        if (chimera):
            name = "%s.embedding%05i_chimera%03i_%03i_%03i" % (instancefile, e, chimera['m'], chimera['n'], chimera['t'])
        embedfile = "%s.yaml" % name
        if not os.path.exists(embedfile) or not use_snapshots:
            print "Calculate embedding ..."
            s.calculateEmbedding(eIndex=e, verbose=verbose, **embedparams)
            s.writeEmbedding(embedfile, eIndex=e)
        else:
            print "Read in embedding ..."
            s.readEmbedding(embedfile, eIndex=e)
        NPhysQubits = len([item for sublist in s.embeddings[e] for item in sublist])
        print "Number of physical Qubits: %i" % NPhysQubits
        if not any(s.embeddings.values()) and e >= retry_embedding:
            if retry_embedding_desperate:
                embedparams['max_no_improvement'] = 30
                embedparams['timeout'] = 500
                embedparams['tries'] = 50
                retry_embedding *= 2
                retry_embedding_desperate = False
            else:
                print "Last %i embeddings were unsuccessfull. Skip the other ones" % (retry_embedding + 1)
                break

        if not embedding_only and NPhysQubits:
            ###################################
            # solve problem
            ###################################
            num_reads = 10000
            physRawSolutionFile = "%s.physRawSolutions.npy" % name
            logRawSolutionFile = "%s.logRawSolutions.npy" % name
            energiesFile = "%s.energies.npy" % name
            numOccurrencesFile = "%s.numOccurrences.npy" % name
            if not os.path.exists(physRawSolutionFile) or not os.path.exists(logRawSolutionFile) or not os.path.exists(energiesFile) or not os.path.exists(numOccurrencesFile) or not use_snapshots:
                print "Calculate solutions ..."
                physRawResult, logRawResult, energies, numOccurrences = s.solve(num_reads=num_reads, eIndex=e)
                np.save(physRawSolutionFile, physRawResult)
                np.save(logRawSolutionFile, logRawResult)
                np.save(energiesFile, energies)
                np.save(numOccurrencesFile, numOccurrences)
            else:
                print "Read in solution ..."
                physRawResult = np.load(physRawSolutionFile)
                logRawResult = np.load(logRawSolutionFile)
                energies = np.load(energiesFile)
                numOccurrences = np.load(numOccurrencesFile)
            print "Solution has energy: %f" % energies[0]
            for k, v in subqubos.items():
                print "Contribution of %s term: %f" % (k, v.evaluate(logRawResult[0]))

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

    if exact:
        ###################################
        # solve problem exactly
        ###################################
        name = "%s" % instancefile
        rawExactSolutionFile = "%s.rawExactSolution.yaml" % name
        if not os.path.exists(rawExactSolutionFile) or not use_snapshots:
            print "Calculate exact solution ..."
            rawresult = s.solve_exact(timeout=timeout)
            if not rawresult:
                print "No exact solution found. Timeout was ", timeout, "seconds"
                return
            f = open(rawExactSolutionFile, 'w')
            yaml.dump(rawresult, f)
            f.close()
        else:
            print "Read in exact solution ..."
            f = open(rawExactSolutionFile, 'r')
            rawresult = yaml.load(f)
            f.close()
        print "Exact solution has energy: %f" % rawresult['energy']
        for k, v in subqubos.items():
            print "Contribution of %s term: %f" % (k, v.evaluate(rawresult['solution']))

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

if __name__ == "__main__":
    main()
