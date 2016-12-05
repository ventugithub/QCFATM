#!/usr/bin/env python
import argparse
import os
from solveInstanceCP import solve_instances

def main():
    parser = argparse.ArgumentParser(description='Solve multiple ATM instances with constraint programming optimization solver',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', default='data/results/', help='result folder')
    parser.add_argument('-d', '--maxDelay', default=18, help='Maximum delay', type=int)
    parser.add_argument('-n', '--numDelays', nargs='+', help='List of (number of delay steps - 1) (ignored if allfactors is true)', type=int)
    parser.add_argument('--allfactors', action='store_true', help='Use (number of delay steps - 1) = all factors of maxDelay ')
    parser.add_argument('--instanceFolder', default='data/instances', help='path to instance folder')
    parser.add_argument('--deltat', default=3, help='Temporal threshold for conflicts', type=int)
    parser.add_argument('--np', default=1, help='Number of processes', type=int)
    parser.add_argument('--use_snapshots', action='store_true', help='use snapshot files')
    parser.add_argument('--verbose', action='store_true', help='verbose output')
    parser.add_argument('--timeout', default=None, help='timeout in seconds for exact solver', type=int)
    parser.add_argument('--inventory', default='data/inventory.h5', help='Inventory file')
    parser.add_argument('--pmin', default=0, help='minimum index of partition to consider', type=int)
    parser.add_argument('--pmax', default=113, help='maximum index of partition to consider', type=int)
    parser.add_argument('--nmin', default=3, help='minimum number of delay steps - 1 (ignored if numDelays is given)', type=int)
    parser.add_argument('--nmax', default=18, help='maximum number of delay steps - 1 (ignored if numDelays is given)', type=int)
    args = parser.parse_args()

    if args.allfactors:
        numDelays = []
        for n in range(1, args.maxDelay + 1):
            if args.maxDelay % n == 0:
                numDelays.append(n)
    elif args.numDelays:
        numDelays = args.numDelays
    else:
        numDelays = range(args.nmin, args.nmax + 1)
    partitions = range(args.pmin, args.pmax + 1)
    print "Collect instance files ..."
    instancefiles = []
    for p in partitions:
        instancefile = '%s/atm_instance_partition%04i_delayStep009_maxDelay018.h5' % (args.instanceFolder, p)
        if not os.path.exists(instancefile):
            raise ValueError('%s does not exists' % instancefile)
        instancefiles.append(instancefile)
    print "Solve instances ..."
    solve_instances(instancefiles=instancefiles,
                    numDelays=numDelays,
                    maxDelay=args.maxDelay,
                    deltat=args.deltat,
                    np=args.np,
                    outputFolder=args.output,
                    use_snapshots=args.use_snapshots,
                    verbose=args.verbose,
                    timeout=args.timeout,
                    inventoryfile=args.inventory)

if __name__ == "__main__":
    main()
