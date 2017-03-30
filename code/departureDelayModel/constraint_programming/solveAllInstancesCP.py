#!/usr/bin/env python
import os
import subprocess
from solveInstanceCP import solve_instances

nProcs = 64
inventory = 'data/connectedComponents/inventory.h5'
instancefiles = []
outputFolders = {}
maxDelayDict = {}
numDelayDict  = {}
# the maximum delay of the precalculation
for maxDelayPrecalc in [6, 9, 12, 15, 18, 24, 36, 48, 60]:
    instanceFolder = '../../../precalculation/data/instances/connectedComponents/maxDepartDelayPrecalculation_%03i' % maxDelayPrecalc
    outputFolder = 'data/connectedComponents/maxDelayPrecalc%03i/results' % maxDelayPrecalc
    # get partition maximum
    cmd = 'ls %s | tail -n 1' % instanceFolder
    return_string = subprocess.check_output(cmd, shell=True)
    partitionMax = int(return_string.partition('partition')[2].split('_')[0])
    maxDelayVariables = []
    numDelays = {}
    for n in range(1, maxDelayPrecalc + 1):
        if maxDelayPrecalc % n == 0:
            maxDelayVariables.append(maxDelayPrecalc / n)
    for maxDelayVariable in maxDelayVariables:
        numDelayList = []
        for n in range(1, maxDelayVariable + 1):
            if maxDelayVariable % n == 0:
                numDelayList.append(n)
        numDelays[maxDelayVariable] = numDelayList

    for p in range(partitionMax + 1):
        instancefile = '%s/atm_instance_partition%04i_delayStep%03i_maxDelay%03i.h5' % (instanceFolder, p, maxDelayPrecalc, maxDelayPrecalc)
        if not os.path.exists(instancefile):
            raise ValueError('%s does not exists' % instancefile)
        instancefiles.append(instancefile)
        outputFolders[instancefile] = outputFolder
        maxDelayDict[instancefile] = maxDelayVariables
        for maxDelayVariable in maxDelayVariables:
            numDelayDict[(instancefile, maxDelayVariable)] = numDelays[maxDelayVariable]


solve_instances(instancefiles,
                outputFolders,
                maxDelayDict,
                numDelayDict,
                np=nProcs,
                use_snapshots=True,
                skipBigProblems=5000,
                verbose=False,
                timeout=1000,
                inventoryfile=inventory)

