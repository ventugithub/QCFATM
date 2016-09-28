#!/usr/bin/env python
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='copy data from pleiades cluster')
    parser.add_argument('-u', '--user', default='tstollen', help='user name on cluster')
    parser.add_argument('-n', '--host', default='pfe23', help='host name on cluster')
    parser.add_argument('-p', '--path', default='projects/qcfatm/code/departureDelayModel/qubo/data/', help='path to data folder on cluster')
    parser.add_argument('-d', '--destination', default='data', help='path to destination data folder on local machine')
    parser.add_argument('-e', '--exclude', default='*.npy', help='exclude string (e.g. "*.npy", argument to rsync exclude option)')
    args = parser.parse_args()

    cmd = "rsync -avz --exclude '%s' %s@%s.nas.nasa.gov:%s %s" % (args.exclude, args.user, args.host, args.path, args.destination)
    subprocess.call(cmd, shell=True)
if __name__ == "__main__":
    main()
