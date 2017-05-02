#!/usr/bin/env python
import os
import argparse

from instance import Instance
from variable import Variable
from qubo import Qubo
from qcfco.validityBoundaryTracker import ParallelValidityBoundaryTracker


def main():
    parser = argparse.ArgumentParser(description='Solve departure only model exactly and scan for threshold in penalty weights at which the solutions become invalid',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', default='data/partitions/results', help='result folder')
    parser.add_argument('--instanceFolder', default='data/partitions/instances', help='path to instance folder')
    parser.add_argument('--inventoryfile', default='data/partitions/analysis/inventory.h5', help='inventory file')
    parser.add_argument('--pmin', default=0, help='minimum index of partition to consider', type=int)
    parser.add_argument('--pmax', default=79, help='maximum index of partition to consider', type=int)
    parser.add_argument('-d', '--delays', nargs='+', default=[3], help='delay steps to consider', type=int)
    parser.add_argument('--use_snapshots', action='store_true', help='use snapshot files')
    parser.add_argument('--timeout', default=1000, help='timeout in seconds for exact solver')
    parser.add_argument('--maxDelay', default=18, help='maximum delay', type=int)
    parser.add_argument('--skipBigProblems', help='Number of logical qubits above which no calculation is performed', type=int)
    parser.add_argument('--delta_w', default=0.01, help='accuray of the bisection algorithm (rounded to 3 digits)', type=float)
    parser.add_argument('--wfixed1', default=2.0, help='fixed first penalty weight of the QUBO for first bisection search of threshold', type=float)
    parser.add_argument('--wstart1', default=0.001, help='starting value of first penalty weight for first bisection search of threshold (rounded to 3 digits)', type=float)
    parser.add_argument('--counter_clockwise', action='store_true', help='Counter clockwise search for threshold points on a circle during the algorithm following the threshold boundary')
    parser.add_argument('--max_penalty_1', default=2.5, help='upper bound for first penalty weights', type=float)
    parser.add_argument('--max_penalty_2', default=2.5, help='upper bound for second penalty weights', type=float)
    parser.add_argument('--maxIter', default=200, help='maximum number of circular bisection steps', type=int)
    parser.add_argument('--radiuses', nargs='+', default=[0.1, 0.2, 0.5, 1], help='radiuses used in algorithm following the threshold boundary', type=float)
    parser.add_argument('--np', default=1, help='number of processes', type=int)
    args = parser.parse_args()

    partitions = range(args.pmin, args.pmax + 1)

    trackValidityBoundaryArgs = {'outputFolder': args.output,
                                 'use_snapshots': args.use_snapshots,
                                 'timeout': args.timeout,
                                 'skipBigProblems': args.skipBigProblems,
                                 'wstart1': args.wstart1,
                                 'delta_w': args.delta_w,
                                 'wfixed1': args.wfixed1,
                                 'counter_clockwise': args.counter_clockwise,
                                 'max_penalty_1': args.max_penalty_1,
                                 'max_penalty_2': args.max_penalty_2,
                                 'radiuses': args.radiuses,
                                 'inventoryfile': args.inventoryfile}

    print "Collect instance files ..."
    instancefiles = []
    for d in args.delays:
        for p in partitions:
            instancefile = '%s/atm_instance_partition%04i_delayStep%03i_maxDelay%03i.h5' % (args.instanceFolder, p, d, args.maxDelay)
            if not os.path.exists(instancefile):
                raise ValueError('%s does not exists' % instancefile)
            instancefiles.append(instancefile)
    print "Solve instances ..."
    parallelTracker = ParallelValidityBoundaryTracker(Instance, Variable, Qubo)
    parallelTracker.trackValidityBoundary(instancefiles, np=args.np, **trackValidityBoundaryArgs)

if __name__ == "__main__":
    main()
