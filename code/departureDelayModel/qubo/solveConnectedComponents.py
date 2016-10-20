#!/usr/bin/env python
import glob
import solveInstance as si

delays = [3, 6, 9]
penalty_weights = [0.5, 1, 2]
num_embed = 5
inventoryfile = 'data/instances/inventory.csv'
partitions = range(0, 96)
timeout = 1000
np = 1

print "Collect instancefiles ..."
instancefiles = {}
for d in delays:
    for p in partitions:
        files = glob.glob('data/instances/instances_d%i/atm_instance_partition%04i_f????_c?????.yaml' % (d, p))
        assert len(files) == 1
        instancefiles[(d, p)] = files[0]
print "Solve instances ..."
for w in penalty_weights:
    w2 = w
    w3 = w
    solve_instance_args = {'num_embed': num_embed,
                           'use_snapshots': True,
                           'retry_embedding': max(num_embed - 2, 0),
                           'retry_embedding_desperate': 1,
                           'unary': True,
                           'verbose': False,
                           'timeout': timeout,
                           'exact': True,
                           'store_everything': True,
                           'retry_exact': False,
                           'inventoryfile': inventoryfile}
    si.solve_instances(instancefiles.values(), penalty_weights={'unique': w2, 'conflict': w3}, np=np, **solve_instance_args)
