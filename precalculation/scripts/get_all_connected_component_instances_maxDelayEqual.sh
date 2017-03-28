#!/bin/bash
for maxDelay in 6 9 12 15 18 24 36 48 60
do
    t=$(($maxDelay + 3))
    maxDelayStr=$(printf "%03d" $maxDelay)
    ./get_instances.py -t $t --maxDepartDelay $maxDelay --maxDelays $maxDelay --delayStepsAllFactors --output data/instances/connectedComponents/maxDepartDelayPrecalculation_$maxDelayStr/ connectedComponents
done

