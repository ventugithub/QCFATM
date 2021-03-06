#!/usr/bin/env python
import argparse
import os

import solveInstance as si
from create_random_instances import create_instances as ci

def main():
    parser = argparse.ArgumentParser(description='Create NASIC instances', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', default='data/random_instances/instances', help='output folder for instances')
    parser.add_argument('--outputResults', default='data/random_instances/results', help='output folder for results')
    parser.add_argument('-n', '--repetitions', default='10', help='number of repetitions', type=int)
    parser.add_argument('-f', '--Fmin', default='10', help='Minimum number of flights', type=int)
    parser.add_argument('-F', '--Fmax', default='10', help='Maximum number of flights', type=int)
    parser.add_argument('-c', '--Cmin', default='4', help='Minimum number of conflicts', type=int)
    parser.add_argument('-C', '--Cmax', default='4', help='Max number of conflicts', type=int)
    parser.add_argument('--Tmin', default='100', help='Minimum value of total time range', type=int)
    parser.add_argument('--Tmax', default='100', help='Maximum value of total time range ', type=int)
    parser.add_argument('--tmin', default='10', help='Minimum value of sigma in arrival time at conflict', type=int)
    parser.add_argument('--tmax', default='10', help='Maximum value of sigma in arrival time at conflict', type=int)
    parser.add_argument('--Dmin', default='18', help='Minimum value maximal delays', type=int)
    parser.add_argument('--Dmax', default='18', help='Maximum value maximal delays', type=int)
    parser.add_argument('--dmin', default='3', help='Minimum value delay steps', type=int)
    parser.add_argument('--dmax', default='3', help='Maximum value delay steps ', type=int)

    parser.add_argument('--num_embed', default=1, help='number of different embeddings', type=int)
    parser.add_argument('--use_snapshots', action='store_true', help='use snapshot files')
    parser.add_argument('--qubo_creation_only', action='store_true', help='qubo creation only')
    parser.add_argument('--embedding_only', action='store_true', help='no quantum annealing')
    parser.add_argument('--retry_embedding', default=0, help='Number of retrys after embedding failed', type=int)
    parser.add_argument('--retry_embedding_desperate', action='store_true', help='try extreme values for embedding')
    parser.add_argument('--retry_exact', action='store_true', help='retry exact solution in case of previous failure')
    parser.add_argument('--binary', action='store_true', help='Use binary representation of integer variables instead of unary representation')
    parser.add_argument('--verbose', action='store_true', help='verbose output')
    parser.add_argument('--timeout', default=None, help='timeout in seconds for exact solver')
    parser.add_argument('--chimera_m', default=None, help='Number of rows in Chimera', type=int)
    parser.add_argument('--chimera_n', default=None, help='Number of columns in Chimera', type=int)
    parser.add_argument('--chimera_t', default=None, help='Half number of qubits in unit cell of Chimera', type=int)
    parser.add_argument('--exact', action='store_true', help='calculate exact solution with maxsat solver')
    parser.add_argument('--store_everything', action='store_true', help='store everything (e.g. physical raw solution)')
    parser.add_argument('--inventory', default='data/random_instances/analysis/inventory.h5', help='Inventory file')
    parser.add_argument('-p2', '--penalty_weight_unique', default=1, help='penaly weight for the term in the QUBO which enforces uniqueness', type=float)
    parser.add_argument('-p3', '--penalty_weight_conflict', default=1, help='penaly weight for the conflict term in the QUBO', type=float)

    parser.add_argument('-p', '--np', default=1, help='number of parallel processes', type=int)
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
    if (args.np != 1 and not args.embedding_only):
        parser.error('You can run in parallel only if the --embedding_only option is set')
    penalty_weights = {
        'unique': args.penalty_weight_unique,
        'conflict': args.penalty_weight_conflict
    }

    # create output folders
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.path.exists(os.path.dirname(args.inventory)):
        os.makedirs(os.path.dirname(args.inventory))

    filenames = ci(output=args.output,
                   repetitions=args.repetitions,
                   Fmin=args.Fmin, Fmax=args.Fmax,
                   Cmin=args.Cmin, Cmax=args.Cmax,
                   Tmin=args.Tmin, Tmax=args.Tmax,
                   tmin=args.tmin, tmax=args.tmax,
                   Dmin=args.Dmin, Dmax=args.Dmax,
                   dmin=args.dmin, dmax=args.dmax)

    solve_instance_args = {'num_embed': args.num_embed,
                           'outputFolder': args.outputResults,
                           'use_snapshots': args.use_snapshots,
                           'qubo_creation_only': args.qubo_creation_only,
                           'embedding_only': args.embedding_only,
                           'retry_embedding': args.retry_embedding,
                           'retry_embedding_desperate': args.retry_embedding_desperate,
                           'unary': not args.binary,
                           'verbose': args.verbose,
                           'timeout': args.timeout,
                           'chimera': chimera,
                           'exact': args.exact,
                           'store_everything': args.store_everything,
                           'retry_exact': args.retry_exact,
                           'inventoryfile': args.inventory}
    si.solve_instances(filenames, penalty_weights, np=args.np, **solve_instance_args)

if __name__ == "__main__":
    main()
