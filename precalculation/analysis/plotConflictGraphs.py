#!/usr/bin/env python
# from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.backends.backend_pdf
import progressbar
import subprocess
import argparse
import sys
sys.path.append('../')
import analysis


def main():
    parser = argparse.ArgumentParser(description='Plot conflicts in real space to a PDF file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', default='../data/TrajDataV2_20120729.txt', help='input file containing the trajectory data with consecutive flight index')
    parser.add_argument('-d', '--mindistance', default=30, help='Minimum distance in nautic miles to qualify as a conflict', type=float)
    parser.add_argument('-t', '--mintime', default=21, help='Minimum time difference in minutes to qualify as a potential conflict', type=int)
    parser.add_argument('--delayPerConflict', default=0, help='Delay introduced by each conflict avoiding maneuver', type=int)
    parser.add_argument('--dthreshold', default=3, help='Minimum time difference in minutes to qualify as a real conflict', type=int)
    parser.add_argument('--maxDepartDelay', default=18, help='Maximum departure delay', type=int)
    parser.add_argument('--kmin', default=0, help='Minimum conflict index to plot', type=int)
    parser.add_argument('--kmax', default=None, help='Maximum conflict index to plot (default: maximum available)', type=int)
    parser.add_argument('--ncols', default=1, help='Number of columns', type=int)
    parser.add_argument('--nrows', default=3, help='Number of rows per page', type=int)
    parser.add_argument('--output', default='conflictGraphs.pdf', help='Output PDF file name')
    args = parser.parse_args()

    mindistance = args.mindistance
    mintime = args.mintime
    delay = args.delayPerConflict
    dthreshold = args.dthreshold
    maxDepartDelay = args.maxDepartDelay
    inputFile = args.input
    name = "mindist%05.1f_mintime%03i" % (mindistance, mintime)
    reducedPointConflictFile = "%s.%s.reducedPointConflicts_delay%03i_thres%03i_depart%03i.csv" % (inputFile, name, delay, dthreshold, maxDepartDelay)
    reducedParallelConflictFile = "%s.%s.reducedParallelConflicts_delay%03i_thres%03i_depart%03i.csv" % (inputFile, name, delay, dthreshold, maxDepartDelay)
    reducedPointConflicts = pd.read_csv(reducedPointConflictFile, index_col='conflictIndex')
    reducedParallelConflicts = pd.read_csv(reducedParallelConflictFile, index_col='parallelConflict')

    title = 'Connected components of the conflict graph'
    repoversion = subprocess.check_output(['git', 'rev-parse', 'HEAD']).rstrip('\n')
    description = """
    Generated with plotConflictGraphs.py
    Version: %s
    Parameters:
    - mintime = %i
    - mindistance = %i
    - delay = %i
    - dthreshold = %i
    - maxDepartDelay = %i
    """ % (repoversion, mintime, mindistance, delay, dthreshold, maxDepartDelay)

    nperpage = args.nrows
    Ncols = args.ncols

    G = analysis.getConflictGraph(reducedPointConflicts, reducedParallelConflicts)
    components = nx.connected_component_subgraphs(G)
    sortedComponents = sorted(list(components), key=lambda x: len(x.nodes()))
    Nc = len(sortedComponents)
    pdf = matplotlib.backends.backend_pdf.PdfPages(args.output)
    pbar = progressbar.ProgressBar().start()
    pbar.maxval = Nc

    for n in range(Nc):
        if n % (nperpage * Ncols) == 0:
            fig = plt.figure(figsize=(10 * Ncols, 6 * nperpage))
            fig.suptitle(title, fontsize=14, fontweight='bold')
            fig.text(.1, .1, description)
        ax = fig.add_subplot(nperpage + 1, Ncols, n % (nperpage * Ncols) + 1)
        analysis.plotConflictGraph(G, component=n, ax=ax)
        if n % (nperpage * Ncols) == (nperpage * Ncols) - 1 or n == Nc - 1:
            pdf.attach_note('Conflict Graphs')
            pdf.savefig(figure=fig)
        pbar.update(n)
    pbar.finish()
    pdf.close()

if __name__ == "__main__":
    main()
