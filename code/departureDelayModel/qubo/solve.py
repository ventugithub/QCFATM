#!/usr/bin/env python

from instance import Instance
from variable import Variable
from qubo import Qubo
from qcfco.controller import Controller


def main():
    # get controller instance
    controller = Controller(Instance, Variable, Qubo)
    # get command line arguments for solve instance function of the controller class
    parser = controller.getParser()
    # add command line arguments for penalty weights
    subqubonames = Qubo.get_subqubo_names()
    for subquboname in subqubonames:
        parser.add_argument('--pw_%s' % subquboname, default='1', help='Penalty weight for "%s" term in QUBO' % subquboname, type=float)
    # parse command line arguments
    args = parser.parse_args()

    # get penalty weights dictionary
    argdict = vars(args)
    penalty_weights = {}
    for subquboname in subqubonames:
        pw = argdict['pw_%s' % subquboname]
        penalty_weights[subquboname] = pw
    print penalty_weights

    # solve instance
    controller.solve_instance(instancefile=args.input,
                              outputFolder=args.output,
                              penalty_weights=penalty_weights,
                              num_embed=args.num_embed,
                              use_snapshots=args.use_snapshots,
                              embedding_only=args.embedding_only,
                              qubo_creation_only=args.qubo_creation_only,
                              retry_embedding=args.retry_embedding,
                              retry_embedding_desperate=args.retry_embedding_desperate,
                              verbose=args.verbose,
                              timeout=args.timeout,
                              exact=args.exact,
                              chimera_m=args.chimera_m,
                              chimera_n=args.chimera_n,
                              chimera_t=args.chimera_t,
                              inventoryfile=args.inventory,
                              store_everything=args.store_everything,
                              retry_exact=args.retry_exact,
                              solverConfig=args.solverConfig)

if __name__ == "__main__":
    main()
