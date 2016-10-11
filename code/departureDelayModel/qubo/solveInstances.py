#!/usr/bin/env python
import subprocess
import itertools
import glob

delays = [3, 6, 9]
penalty_weights = [0.5, 1, 2]
num_embed = 5
inventoryfile = 'data/instances/inventory.csv'
partitions = range(0, 80)

for (d, w) in itertools.product(delays, penalty_weights):
    for p in partitions:
        instancefiles = glob.glob('data/instances/instances_d%i/atm_instance_partition%04i_f????_c?????.yaml' % (d, p))
        assert len(instancefiles) == 1
        cmd = './runInstance.py --num_embed %i -i %s --exact --unary  --use_snapshots --inventory %s -p2 %f -p3 %f' % (num_embed, instancefiles[0], inventoryfile, w, w)
        print cmd
        subprocess.call(cmd, shell=True)
