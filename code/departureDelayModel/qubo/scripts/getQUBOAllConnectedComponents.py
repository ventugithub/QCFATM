#!/usr/bin/env python
import subprocess

skipBigProblems = 1024
num_embed = 5
nProc = 1
# the maximum delay of the precalculation
for maxDelayPrecalc in [6, 12, 15, 18, 24, 36, 48, 60]:
    # get the maximum value of the delay variable (always <= the maximum delay of the precalculation)
    instanceFolder = '../../../precalculation/data/instances/connectedComponents/maxDepartDelayPrecalculation_%03i' % maxDelayPrecalc
    # get partition maximum
    cmd = 'ls %s | tail -n 1' % instanceFolder
    return_string = subprocess.check_output(cmd, shell=True)
    partitionMax = int(return_string.partition('partition')[2].split('_')[0])
    delaySteps = []
    for n in range(1, maxDelayPrecalc + 1):
        if maxDelayPrecalc % n == 0:
            delaySteps.append(str(maxDelayPrecalc / n))

    cmd = ''
    cmd += './solveConnectedComponents.py'
    cmd += ' --inventoryfile data/connectedComponents/maxDelayPrecalc%03i/inventory.h5' % maxDelayPrecalc
    cmd += ' -o data/connectedComponents/maxDelayPrecalc%03i/results' % maxDelayPrecalc
    cmd += ' --instanceFolder %s/' % instanceFolder
    cmd += ' --pmax %i' % partitionMax
    cmd += ' --num_embed %i' % num_embed
    cmd += ' -d %s' % ' '.join(delaySteps)
    # cmd += ' --use_snapshots'
    cmd += ' --qubo_creation_only'
    cmd += ' --maxDelay %i' % maxDelayPrecalc
    cmd += ' --skipBigProblems %i' % skipBigProblems
    cmd += ' --penalty_weights_two_tuples 1 1'
    cmd += ' --np %i' % nProc
    print cmd
    subprocess.call(cmd, shell=True)
