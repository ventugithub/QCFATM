#!/usr/bin/env python
import subprocess

# the maximum delay of the precalculation
for maxDelayPrecalc in [6, 12, 15, 18, 24, 36, 48, 60]:
    # get the maximum value of the delay variable (always <= the maximum delay of the precalculation)
    instanceFolder = '../../../precalculation/data/instances/connectedComponents/maxDepartDelayPrecalculation_%03i' % maxDelayPrecalc
    # get partition maximum
    cmd = 'ls %s | tail -n 1' % instanceFolder
    return_string = subprocess.check_output(cmd, shell=True)
    partitionMax = int(return_string.partition('partition')[2].split('_')[0])
    maxDelayVariables = []
    for n in range(1, maxDelayPrecalc + 1):
        if maxDelayPrecalc % n == 0:
            maxDelayVariables.append(maxDelayPrecalc / n)
    mintime = maxDelayPrecalc + 3

    for maxDelayVariable in maxDelayVariables:
        cmd = ''
        cmd += './solveMultipleInstancesCP.py'
        cmd += ' -o data/connectedComponents/maxDelayPrecalc%03i/results' % maxDelayPrecalc
        cmd += ' --instanceFolder %s/' % instanceFolder
        cmd += ' -d %i' % maxDelayVariable
        cmd += ' --allfactors'
        cmd += ' --use_snapshots'
        cmd += ' --inventory data/connectedComponents/maxDelayPrecalc%03i/inventory.h5' % maxDelayPrecalc
        cmd += ' --pmax %i' % partitionMax
        cmd += ' --delayStepInstance %i' % maxDelayPrecalc
        cmd += ' --maxDelayInstance %i' % maxDelayPrecalc
        cmd += ' --timeout %i' % 1
        print cmd
        subprocess.call(cmd, shell=True)

