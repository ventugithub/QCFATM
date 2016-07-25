import os
import yaml
import numpy as np
import dwave_sapi2.remote as remote
import dwave_sapi2.embedding as embedding
import dwave_sapi2.util as util
import dwave_sapi2.core as core
import myToken

import libquail.exact_solvers.qubo_via_maxsat as qubo_via_maxsat

def get_chimera_adjacency(m, n, t):
    return util.get_chimera_adjacency(m, n, t)

class Solver:
    def __init__(self, qubo):
        self.qubo = qubo
        # convert to DWave QUBO format
        self.Q, self.qubo_offset = self.qubo.getDWaveQUBO()
        # conver tot Ising format
        self.h, self.J, self.ising_offset = util.qubo_to_ising(self.Q)

        # connection
        self.conn = None
        # hardware adjacency
        self.hwa = None
        # embedding
        self.embeddings = None

    def establishConnection(self):
        if not self.conn or not self.solver:
            # print "Connect to DWave machine ..."
            # create a remote connection
            try:
                self.conn = remote.RemoteConnection(myToken.myUrl, myToken.myToken)
                # get the solver
                self.solver = self.conn.get_solver('C12')
            except:
                print "Unable to establich connection to %s" % myToken.myUrl
                print "Did you set up the connection?"
                exit(1)

    def calculateHardwareAdjaceny(self):
        self.establishConnection()
        self.hwa = util.get_hardware_adjacency(self.solver)

    def readHardwareAdjaceny(self, filename):
        f = open(filename, 'r')
        hwa_dict = yaml.load(f)
        f.close()
        self.hwa = hwa_dict['hardware_adjacency']

    def getHardwareAdjacency(self, use_snapshots=False, filename='hardware_adjacency.yml'):
        if not self.hwa:
            if not os.path.exists(filename) or not use_snapshots:
                self.calculateHardwareAdjaceny()
                hwa_dict = {'hardware_adjacency': list(self.hwa)}
                f = open(filename, 'w')
                yaml.dump(hwa_dict, f)
                f.close()
            else:
                # print "Read hardware adjacency from file %s" % filename
                self.readHardwareAdjaceny(filename)
        return self.hwa

    def calculateEmbedding(self, eIndex=0, **kwargs):
        if not self.hwa:
            self.getHardwareAdjacency(use_snapshots=True)
        # print "Calculate embedding ..."
        if not self.embeddings:
            self.embeddings = {}
        self.embeddings[eIndex] = embedding.find_embedding(self.J, self.hwa, **kwargs)

    def solve(self, annealing_time=20, num_reads=10000, eIndex=0, **kwargs):
        if not self.embeddings or not self.embeddings[eIndex]:
            self.calculateEmbedding(eIndex=eIndex)
        if not self.hwa:
            self.getHardwareAdjacency(use_snapshots=True)
        self.establishConnection()
        h0, j0, jc, new_embed = embedding.embed_problem(self.h, self.J, self.embeddings[eIndex], self.hwa, **kwargs)
        jtot = jc
        jtot.update(j0)
        # print "Annealing ..."
        result = core.solve_ising(self.solver, h0, jtot, annealing_time=annealing_time, num_reads=num_reads)
        unembedded_result = embedding.unembed_answer(result['solutions'], new_embed, 'minimize_energy', self.h, self.J)

        # take the lowest energy solution
        rawsolution_phys = (np.array(result['solutions']) + 1)/2
        rawsolution_log = (np.array(unembedded_result) + 1)/2
        return rawsolution_phys, rawsolution_log, np.array(result['energies']) + self.qubo_offset + self.ising_offset, np.array(result['num_occurrences'])

    def solve_exact(self, timeout=None):
        r = {}
        solution = qubo_via_maxsat.solve_qubo(self.qubo.getDWaveQUBO()[0], self.qubo.getNVariables(), timeout=timeout)
        if solution:
            r['solution'] = solution
            r['energy'] = self.qubo.evaluate(r['solution'])
            return r
        else:
            return None

    def writeEmbedding(self, filename, eIndex=0):
        """Save embedding to yaml file"""
        f = open(filename, 'w')
        yaml.dump(self.embeddings[eIndex], f)
        f.close()

    def readEmbedding(self, filename, eIndex=0):
        """Read embedding to yaml file"""
        if not self.embeddings:
            self.embeddings = {}
        f = open(filename, 'r')
        self.embeddings[eIndex] = yaml.load(f)
        f.close()