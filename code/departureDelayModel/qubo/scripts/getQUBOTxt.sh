#!/bin/bash
# delay step size in minutes
delayStep=3
# partition
partition=10
# penalty weight for unique term in QUBO
p2=1
# penalty weight for conflict term in QUBO
p3=1

# maximum departure delay in minutes
maxDelay=18
# time below to space time points are in temporal conflict
threshold=3


#########################
# precalculation
#########################
cd ../../../../precalculation/
echo "Precalculation ..."
./precalculateConflicts.py -t $(($maxDelay + $threshold)) --maxDepartDelay $maxDelay --use_snapshots

#########################
# extract instances
#########################
echo "Get instances ..."
mkdir -p ../code/departureDelayModel/qubo/data/partitions/instances
./get_instances.py --maxDelay $maxDelay --delayStep $delayStep --maxDepartDelay $maxDelay -t $(($maxDelay + $threshold)) --output ../code/departureDelayModel/qubo/data/partitions/instances/
cd ../code/departureDelayModel/qubo/

#########################
# get qubo
#########################
echo "Get QUBO ..."
instancefile=$(printf 'data/partitions/instances/atm_instance_partition%04d_delayStep%03d_maxDelay%03d.h5' "$partition" "$delayStep" "$maxDelay")
mkdir -p data/partitions/results
touch myToken.py
./solveInstance.py --use_snapshots --qubo_creation_only -p2 $p2 -p3 $p3 -i $instancefile  --unary -o data/partitions/results/

echo "Convert QUBO ..."
#########################
# convert qubo to text format
#########################
resultsfile=$(printf 'data/partitions/results/atm_instance_partition%04d_delayStep%03d_maxDelay%03d.results.h5' "$partition" "$delayStep" "$maxDelay")
mkdir -p data/partitions/qubos/
qubofile=$(printf 'data/partitions/qubos/atm_instance_partition%04d_delayStep%03d_maxDelay%03d_punique%e_pconfict%e.txt' "$partition" "$delayStep" "$maxDelay" "$p2" "$p3")
./convertQUBO2Txt.py --input $resultsfile -o $qubofile -p2 $p2 -p3 $p3
echo "QUBO written to $qubofile"


