#!/usr/bin/env python
import argparse
import os
from solveInstanceCP import solve_instances

def main():
    parser = argparse.ArgumentParser(description='Solve multiple ATM instances with constraint programming optimization solver',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', default='data/results/', help='result folder')
    parser.add_argument('-d', '--maxDelay', default=18, help='Maximum delay', type=int)
    parser.add_argument('-n', '--numDelays', nargs='+', help='List of (number of delay steps - 1). Set to 0 for continuous variables (ignored if allfactors is true)', type=int)
    parser.add_argument('--allfactors', action='store_true', help='Use (number of delay steps - 1) = all factors of maxDelay ')
    parser.add_argument('--instanceFolder', default='data/instances', help='path to instance folder')
    parser.add_argument('--maxDelayInstance', default=18, help='Search for instances of ending with e.g. "_maxDelay006.h5". Note, that the maxDelay of the instance will be overwritten by the maxDelay of the CP solver.', type=int)
    parser.add_argument('--delayStepInstance', default=3, help='Search for instances of containing e.g. "delayStep003_maxDelay006.h5". Note, that the delayStep of the instance will be overwritten by the numDelays of the CP solver.', type=int)
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
    instancefiles = []
    outputFolders = {}
    maxDelayDict = {}
    numDelayDict  = {}
    for p in partitions:
        instancefile = '%s/atm_instance_partition%04i_delayStep%03i_maxDelay%03i.h5' % (args.instanceFolder, p, args.delayStepInstance, args.maxDelayInstance)
        if not os.path.exists(instancefile):
            raise ValueError('%s does not exists' % instancefile)
        instancefiles.append(instancefile)
        outputFolders[instancefile] = args.output
        maxDelayDict[instancefile] = [args.maxDelay]
        numDelayDict[(instancefile, args.maxDelay)] = numDelays

    print "Solve instances ..."
    solve_instances(instancefiles,
                    outputFolders,
                    maxDelayDict,
                    numDelayDict,
                    deltat=args.deltat,
                    np=args.np,
                    use_snapshots=args.use_snapshots,
                    verbose=args.verbose,
                    timeout=args.timeout,
                    inventoryfile=args.inventory)

if __name__ == "__main__":
    main()
