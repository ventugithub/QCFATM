#!/usr/bin/env python
import subprocess
import itertools
import glob
import multiprocessing
from runInstance import atm

delays = [3, 6, 9]
penalty_weights = [0.5, 1, 2]
num_embed = 5
partitions = range(80, 96)

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
        atm_args = {'instancefile': instancefile,
                    'num_embed': num_embed,
                    'use_snapshots': True,
                    'embedding_only': True,
                    'retry_embedding': 5,
                    'retry_embedding_desperate': False,
                    'unary': True,
                    'penalty_weights': penalty_weights,
                    'timeout': 3600,
                    'exact': True}
        pool.apply_async(atm, kwds=atm_args)
pool.close()
pool.join()
