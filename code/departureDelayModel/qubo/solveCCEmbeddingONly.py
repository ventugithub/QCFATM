#!/usr/bin/env python
import itertools
import glob
import multiprocessing
from solveInstanc import solve_instance

delays = [3, 6, 9]
penalty_weights = [0.5, 1, 2]
partitions = range(80, 96)
num_embed = 5
num_embed_desperate = 1

np = 16

pool = multiprocessing.Pool(processes=np)
for (d, w) in itertools.product(delays, penalty_weights):
    for p in partitions:
        instancefiles = glob.glob('data/instances/instances_d%i/atm_instance_partition%04i_f????_c?????.yaml' % (d, p))
        penalty_weights = {
            'conflict': w,
            'unique': w,
        }

        assert len(instancefiles) == 1
        instancefile = instancefiles[0]
        print "Process instance file %s" % instancefile
        solve_instance_args = {'instancefile': instancefile,
                               'num_embed': 5,
                               'use_snapshots': True,
                               'embedding_only': True,
                               'retry_embedding': num_embed - 1 - num_embed_desperate,
                               'retry_embedding_desperate': num_embed_desperate,
                               'unary': True,
                               'penalty_weights': penalty_weights,
                               'timeout': 3600,
                               'exact': True}
        pool.apply_async(solve_instance, kwds=solve_instance_args)
pool.close()
pool.join()
