#!/usr/bin/env python
import subprocess

# the maximum delay of the precalculation
# for maxDelayPrecalc in [12, 18, 24, 36, 48, 60]:
for maxDelayPrecalc in [18, ]:
    # get the maximum value of the delay variable (always <= the maximum delay of the precalculation)
    maxDelayVariables = []
    for n in [6, 12, 18, 24, 36, 38, 60]:
        if n <= maxDelayPrecalc:
            maxDelayVariables.append(str(n))
    mintime = maxDelayPrecalc + 3
    outFolder = './data/instances/connectedComponents/maxDepartDelayPrecalculation_%03i/' % maxDelayPrecalc
    cmd = "./get_instances.py -t %i --output %s --maxDepartDelay %i --maxDelays %s --delayStepsAllFactors connectedComponents" % (mintime, outFolder, maxDelayPrecalc, ' '.join(maxDelayVariables))
    print cmd
    subprocess.call(cmd, shell=True)
