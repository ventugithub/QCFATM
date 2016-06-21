#!/usr/bin/env python
import argparse
import itertools as it
import numpy as np
import yaml
import os
import progressbar

def instance2Yaml(filename, I, J, L, b, m, r, A, B):
    data = {}
    data['I'] = I
    data['J'] = J
    data['L'] = L
    data['b'] = b.tolist()
    data['m'] = m.tolist()
    data['r'] = r.tolist()
    data['A'] = A.tolist()
    data['B'] = B.tolist()
    f = open(filename, 'w')
    yaml.dump(data, f)
    f.close()

def main():
    parser = argparse.ArgumentParser(description='Create NASIC instances', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', default='instances', help='output folder')
    parser.add_argument('-n', '--repetitions', default='10', help='number of repetitions', type=int)
    parser.add_argument('-i', '--Imin', default='3', help='Minimum number of airfields', type=int)
    parser.add_argument('-I', '--Imax', default='5', help='Maximum number of airfields', type=int)
    parser.add_argument('-j', '--Jmin', default='2', help='Minimum number of tasks in units of the number of airfields', type=int)
    parser.add_argument('-J', '--Jmax', default='3', help='Max number of tasks in units of the number of airfields', type=int)
    parser.add_argument('-l', '--Lmin', default='1', help='Minimum number of aircraft types', type=int)
    parser.add_argument('-L', '--Lmax', default='3', help='Maximum number of aircraft types', type=int)
    parser.add_argument('-m', '--Mmin', default='1', help='Minimum number aircrafts of type l needed for a task j', type=int)
    parser.add_argument('-M', '--Mmax', default='5', help='Maximum number aircrafts of type l needed for a task j', type=int)
    parser.add_argument('-b', '--Bmin', default='1', help='Minimum number aircrafts available on a given airfield i', type=int)
    parser.add_argument('-B', '--Bmax', default='10', help='Maximum number aircrafts available on a given airfield i', type=int)
    parser.add_argument('-r', '--Rmin', default='1', help='Minimum number cover ratio', type=int)
    parser.add_argument('-R', '--Rmax', default='3', help='Maximum number cover ratio', type=int)
    args = parser.parse_args()

    Ivalues = range(args.Imin, args.Imax + 1)
    Jvalues = range(args.Jmin, args.Jmax + 1)
    Lvalues = range(args.Lmin, args.Lmax + 1)
    Mmin = args.Mmin
    Mmax = args.Mmax
    Bmin = args.Bmin
    Bmax = args.Bmax
    Rmin = args.Rmin
    Rmax = args.Rmax
    N = args.repetitions
    output = args.output

    if not os.path.exists(output):
        os.mkdir(output)

    # set random seed for reproducibility
    np.random.seed(0)

    print 'Calculate instances'
    # init progress bar
    pbar = progressbar.ProgressBar().start()
    pbar.maxval = len(Ivalues) * len(Jvalues) * len(Lvalues) * N
    count = 0
    for I, J, L in it.product(Ivalues, Jvalues, Lvalues):
        J = J * I
        for n in range(N):
            b = np.random.randint(Bmin, Bmax, (I, L))
            m = np.random.randint(Mmin, Mmax, (L, J))
            r = np.random.randint(Rmin, Rmax, (L, J, L))
            # get A and B
            B = np.sum(b, axis=1)
            A = np.sum(b, axis=0)
            filename = "%s/nasic_instance_I%02i_J%02i_L%02i_M%02i-%02i_B%02i-%02i_R%02i-%02i_n%05i.yml" % (output, I, J, L, Mmin, Mmax, Bmin, Bmax, Rmin, Rmax, n)
            instance2Yaml(filename, I, J, L, b, m, r, B, A)

            # progress bar
            if count % 100 == 0:
                pbar.update(count)
            count = count + 1
    # close progress bar
    pbar.finish()

if __name__ == "__main__":
    main()
