#!/usr/bin/env python
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='copy data from pleiades cluster',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-u', '--user', default='tstollen', help='user name on cluster')
    parser.add_argument('-n', '--host', default='pfe23', help='host name on cluster')
    parser.add_argument('-p', '--path', default='projects/qcfatm/code/departureDelayModel/qubo/data/', help='path to data folder on cluster')
    parser.add_argument('-d', '--destination', default='data', help='path to destination data folder on local machine')
    args = parser.parse_args()

    cmd = "rsync -avz %s@%s:%s %s" % (args.user, args.host, args.path, args.destination)
    print cmd
    subprocess.call(cmd, shell=True)
if __name__ == "__main__":
    main()
