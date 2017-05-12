#!/usr/bin/env python
import argparse
import os
import itertools
from instance import Instance
from variable import Variable
from qubo import Qubo
from qcfco.controller import ParallelController


def main():
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
            pass
    parser = argparse.ArgumentParser(description='Solve departure only model for instances extracted from the connected components of the confict graph',
                                     formatter_class=MyFormatter)
    parser.add_argument('--inventoryfile', default='data/partitions/analysis/inventory.h5', help='inventory file')
    parser.add_argument('-o', '--output', default='data/partitions/results', help='result folder')
    parser.add_argument('--instanceFolder', default='data/partitions/instances', help='path to instance folder')
    parser.add_argument('--pmin', default=0, help='minimum index of partition to consider', type=int)
    parser.add_argument('--pmax', default=79, help='maximum index of partition to consider', type=int)
    parser.add_argument('--num_embed', default=1, help='number of different embeddings', type=int)
    parser.add_argument('-d', '--delays', nargs='+', default=[3, 6, 9], help='delay steps to consider', type=int)
    parser.add_argument('--use_snapshots', action='store_true', help='use snapshot files')
    parser.add_argument('--embedding_only', action='store_true', help='no quantum annealing')
    parser.add_argument('--qubo_creation_only', action='store_true', help='qubo creation only')
    parser.add_argument('--retry_exact', action='store_true', help='retry exact solution in case of previous failure')
    parser.add_argument('--timeout', default=1000, help='timeout in seconds for exact solver')
    parser.add_argument('--num_reads', default=10000, help='Number of D-Wave runs for one call to the D-Wave solver (must be below the hardware restriction', type=int)
    parser.add_argument('--num_repetitions', default=1, help='Number of repeated calls to the D-Wave solver (the total number of annealing runs is num_reads * num_repetitions)', type=int)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--penalty_weights_all_combinations', nargs='+', help='list of penalty weights for unique and conflict term of the QUBO (', type=float)
    group.add_argument('--penalty_weights_two_tuples', nargs='+', default=[0.5, 0.5, 1, 1, 2, 2], help='list of two penalty weights (unique and conflict) of the QUBO (list length must be even). E.g. 0.5 0.5 1 1 2 2', type=float)
    parser.add_argument('--maxDelay', default=18, help='maximum delay', type=int)
    parser.add_argument('--skipBigProblems', help='Number of logical qubits above which no calculation is performed', type=int)
    parser.add_argument('--np', default=1, help='number of processes', type=int)
    parser.add_argument('--jintra', default=-1.0, help='Intra-logical qubit coupling for embedding', type=float)
    solverConfigHelp = """configuration file with the following entries:
                            [DWaveSolver]
                            name=<solvername for D-Wave API>
                            shortname=<shortname>
                            url=<url> (only for remote solvers)
                            token=<token> (only for remote solvers)
                            chimera_m=<chimera_m> (only for virtual solvers)
                            chimera_n=<chimera_n> (only for virtual solvers)
                            chimera_t=<chimera_t> (only for virtual solvers)
                        """
    parser.add_argument('--solverConfig', help=solverConfigHelp)
    # parse command line arguments
    args = parser.parse_args()

    if args.penalty_weights_two_tuples and len(args.penalty_weights_two_tuples) % 2 != 0:
        parser.error('List of penalty weights (for --penalty_weights_two_tuples) must be even')
    if args.penalty_weights_all_combinations:
        penalty_weights = itertools.product(args.penalty_weights_all_combinations, args.penalty_weights_all_combinations)
    else:
        penalty_weights = zip(args.penalty_weights_two_tuples[::2], args.penalty_weights_two_tuples[1::2])

    partitions = range(args.pmin, args.pmax + 1)

    if not os.path.exists(os.path.dirname(args.inventoryfile)):
        os.makedirs(os.path.dirname(args.inventoryfile))

    print "Collect instancefiles ..."
    instancefiles = []
    for d in args.delays:
        for p in partitions:
            instancefile = '%s/atm_instance_partition%04i_delayStep%03i_maxDelay%03i.h5' % (args.instanceFolder, p, d, args.maxDelay)
            if not os.path.exists(instancefile):
                raise ValueError('%s does not exists' % instancefile)
            instancefiles.append(instancefile)

    # get controller instance

    print "Solve instances ..."
    controller = ParallelController(Instance, Variable, Qubo)
    for w2, w3 in penalty_weights:
        solve_instance_args = {'num_embed': args.num_embed,
                               'outputFolder': args.output,
                               'embedding_only': args.embedding_only,
                               'use_snapshots': args.use_snapshots,
                               'retry_embedding': max(args.num_embed - 2, 0),
                               'retry_embedding_desperate': 1,
                               'verbose': False,
                               'timeout': args.timeout,
                               'exact': True,
                               'store_everything': False,
                               'retry_exact': args.retry_exact,
                               'qubo_creation_only': args.qubo_creation_only,
                               'skipBigProblems': args.skipBigProblems,
                               'solverConfig': args.solverConfig,
                               'jintra': args.jintra,
                               'inventoryfile': args.inventoryfile,
                               'num_reads': args.num_reads,
                               'num_repetitions': args.num_repetitions}

        controller.solve_instances(instancefiles, penalty_weights={'departure': 1, 'unique': w2, 'conflict': w3}, np=args.np, **solve_instance_args)

if __name__ == "__main__":
    main()
