#!/bin/bash
for maxDelay in 6 9 12 15 18 24 36 48 60
do
    t=$(($maxDelay + 3))
    ./precalculateConflicts.py -t $t --maxDepartDelay $maxDelay 
done
