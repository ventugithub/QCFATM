#!/usr/bin/env python
import argparse
import itertools as it
import random
import numpy as np
import os
import progressbar

import instance

def main():
    parser = argparse.ArgumentParser(description='Create NASIC instances', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', default='data', help='output folder')
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
    args = parser.parse_args()

    create_instances(output=args.output,
                     repetitions=args.repetitions,
                     Fmin=args.Fmin, Fmax=args.Fmax,
                     Cmin=args.Cmin, Cmax=args.Cmax,
                     Tmin=args.Tmin, Tmax=args.Tmax,
                     tmin=args.tmin, tmax=args.tmax,
                     Dmin=args.Dmin, Dmax=args.Dmax,
                     dmin=args.dmin, dmax=args.dmax)

def create_instances(output='data/',
                     repetitions=10,
                     Fmin=10,
                     Fmax=10,
                     Cmin=4,
                     Cmax=4,
                     Tmin=100,
                     Tmax=100,
                     tmin=10,
                     tmax=10,
                     Dmin=18,
                     Dmax=18,
                     dmin=3,
                     dmax=3,
                     names_only=False):

    FValues = range(Fmin, Fmax + 1)
    CValues = range(Cmin, Cmax + 1)
    TValues = range(Tmin, Tmax + 1)
    tValues = range(tmin, tmax + 1)
    DValues = range(Dmin, Dmax + 1)
    dValues = range(dmin, dmax + 1)
    Nr = repetitions
    output = output

    if not os.path.exists(output):
        os.mkdir(output)

    # set random seed for reproducibility
    random.seed(0)

    filenames = []
    print 'Calculate instances'
    # init progress bar
    pbar = progressbar.ProgressBar().start()
    pbar.maxval = len(FValues) * len(CValues) * len(TValues) * len(tValues) * len(DValues) * len(dValues) * Nr
    count = 0
    for F, C, TRangeMax, TRangeDelta, delayMax, delayDelta in it.product(FValues, CValues, TValues, tValues, DValues, dValues):
        for n in range(Nr):

            flights = [int(f) for f in np.arange(0, F)]
            conflicts = []
            for c in range(C):
                i = flights[random.randint(1, F - 1)]
                j = flights[random.randint(1, F - 1)]
                while (j == i):
                    j = flights[random.randint(1, F - 1)]
                conflicts.append((i, j))
            arrivalTimes = []
            for c in range(C):
                T = random.randint(0 + TRangeDelta, TRangeMax - TRangeDelta)
                Tmin = T - TRangeDelta
                Tmax = T + TRangeDelta
                t1 = random.randint(Tmin, Tmax)
                t2 = random.randint(Tmin, Tmax)
                arrivalTimes.append((t1, t2))
            delays = [int(d) for d in np.arange(0, delayMax + 1, delayDelta)]
            filename = "%s/atm_instance_F%03i_C%03i_T%03i_t%03i_D%03i_d%03i_n%05i.yml" % (output, F, C, TRangeMax, TRangeDelta, delayMax, delayDelta, n)
            if not names_only:
                inst = instance.Instance(flights, conflicts, arrivalTimes, delays)
                inst.save(filename)

            # progress bar
            if count % 100 == 0:
                pbar.update(count)
            count = count + 1

            filenames.append(filename)
    # close progress bar
    pbar.finish()

    return filenames

if __name__ == "__main__":
    main()
