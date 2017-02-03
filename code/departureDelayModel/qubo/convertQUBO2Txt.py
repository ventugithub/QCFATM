#!/usr/bin/env python
import argparse

import polynomial

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input', required=True, help='results file containing QUBO')
    parser.add_argument('-o', '--output', required=True, help='output file for QUBO in txt format')
    parser.add_argument('-p2', '--penalty_weight_unique', default=1, help='penaly weight for the term in the QUBO which enforces uniqueness', type=float)
    parser.add_argument('-p3', '--penalty_weight_conflict', default=1, help='penaly weight for the conflict term in the QUBO', type=float)
    args = parser.parse_args()

    penalty_weights = {
        'unique': args.penalty_weight_unique,
        'conflict': args.penalty_weight_conflict
    }

    # string representing the penalty weights
    pwstr = "pw"
    for k, v in penalty_weights.items():
        pwstr = pwstr + "-%s%0.3f" % (k, v)
    print("Read in QUBO ...")
    q = polynomial.Polynomial()
    q.load_hdf5(args.input, '%s/qubo' % pwstr)
    if not q.isQUBO:
        raise ValueError('Input polynomial is not a QUBO')
    f = open(args.output, 'w')
    for k in sorted(q.poly, key=lambda x: len(x)):
        v = q.poly[k]
        if len(k) == 2:
            f.write("%i %i %e\n" % (k[0], k[1], v))
        elif len(k) == 1:
            f.write("%i %i %e\n" % (k[0], k[0], v))
        elif len(k) == 0:
            f.write("# offset %e\n" % v)
    f.close()


if __name__ == "__main__":
    main()
