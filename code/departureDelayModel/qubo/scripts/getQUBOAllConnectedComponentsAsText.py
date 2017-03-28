#!/usr/bin/env python
import subprocess
import os

import sys
sys.path.append('.')
import polynomial

subqubonames = ['departure', 'unique', 'conflict']
nProc = 1
# the maximum delay of the precalculation
#for maxDelayPrecalc in [6, 12, 15, 18, 24, 36, 48, 60]:
for maxDelayPrecalc in [6, ]:
    # get the maximum value of the delay variable (always <= the maximum delay of the precalculation)
    resultFolder = 'data/connectedComponents/maxDelayPrecalc%03i/results' % maxDelayPrecalc
    outputFolder = 'data/connectedComponents/quboAsTxt/maxDelayPrecalc%03i' % maxDelayPrecalc
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    # get partition maximum
    cmd = 'ls %s | tail -n 1' % resultFolder
    return_string = subprocess.check_output(cmd, shell=True)
    partitionMax = int(return_string.partition('partition')[2].split('_')[0])

    delaySteps = []
    for n in range(1, maxDelayPrecalc + 1):
        if maxDelayPrecalc % n == 0:
            delaySteps.append(maxDelayPrecalc / n)

    print "Collect instancefiles ..."
    resultfiles = []
    for d in delaySteps:
        for p in range(partitionMax + 1):
            resultfile = '%s/atm_instance_partition%04i_delayStep%03i_maxDelay%03i.results.h5' % (resultFolder, p, d, maxDelayPrecalc)
            if not os.path.exists(resultfile):
                raise ValueError('%s does not exists' % resultfile)
            resultfiles.append(resultfile)

    for resultfile in resultfiles:
        print "Convert subqubos in %s to txt" % resultfile
        # string representing the penalty weights
        for sqname in subqubonames:
            q = polynomial.Polynomial()
            q.load_hdf5(resultfile, 'subqubo-%s' % sqname)
            if not q.isQUBO:
                raise ValueError('Input polynomial is not a QUBO')
            txtSubQuboFile = "%s/%s.subqubo_%s.txt" % (outputFolder, os.path.basename(resultfile).rstrip('results.h5'), sqname)
            f = open(txtSubQuboFile, 'w')
            for k in sorted(q.poly, key=lambda x: len(x)):
                v = q.poly[k]
                if len(k) == 2:
                    f.write("%i %i %e\n" % (k[0], k[1], v))
                elif len(k) == 1:
                    f.write("%i %i %e\n" % (k[0], k[0], v))
                elif len(k) == 0:
                    f.write("# offset %e\n" % v)
            f.close()
