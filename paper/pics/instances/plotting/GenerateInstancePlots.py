
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import matplotlib.backends.backend_pdf


# In[2]:

#get_ipython().magic(u'matplotlib inline')


# In[3]:

fs = (5, 4)


# In[4]:

data = pd.read_csv('../data/graph_1.txt', sep='\t')
x = data.index
y = data.values
fig = plt.figure(figsize=fs)
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y, 'o-')
#data.hist(ax=ax, bins=51)
ax.set_xlabel('Number of flights in connected components')
ax.set_ylabel('PDF');
#ax.set_yscale('log')

pdf = matplotlib.backends.backend_pdf.PdfPages('../analysis_cc.pdf');
pdf.savefig(figure=fig);
pdf.close();


# In[7]:

data = pd.read_csv('../data/graph_2.txt', sep='\t', skiprows=1)
x = data.index
y = data.values
fig = plt.figure(figsize=fs)
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y, 'o')
ax.set_xlabel('Degree')
ax.set_ylabel('PDF');

fitx = np.linspace(x.min(), x.max(), 100)
fity = np.power(10, -1.20965-0.0291728 * fitx)
ax.plot(fitx, fity, '--')
ax.set_yscale('log')
pdf = matplotlib.backends.backend_pdf.PdfPages('../connectivity_pdf.pdf');
pdf.savefig(figure=fig);
pdf.close();


# In[8]:

data = pd.read_csv('../data/graph_3.txt', sep='\t', header=None, skiprows=1).values
dmax = data[:, 0]
exponent = data[:, 1]
error = data[:, 2]
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(1, 1, 1)
ax.errorbar(dmax, exponent, yerr=error, fmt='o')
ax.set_xlabel('$d_\mathrm{max}$')
ax.set_ylabel('power law exponent for connectivity');
pdf = matplotlib.backends.backend_pdf.PdfPages('../connectivity_pl.pdf');
pdf.savefig(figure=fig);
pdf.close();


# In[9]:

fig = plt.figure(figsize=fs)
ax = fig.add_subplot(1, 1, 1)

data = pd.read_csv('../data/graph_4a.txt', sep='\t', skiprows=1)
x = data.index
y = data.values
ax.plot(x, y, 'o', label='$d_\mathrm{max}=6$')
fitx = np.linspace(x.min(), x.max(), 100)
fity = 1.36131 + 0.0756234 * fitx
ax.plot(fitx, fity, '--', c=seaborn.xkcd_rgb["denim blue"])

data = pd.read_csv('../data/graph_4b.txt', sep='\t', skiprows=1)
x = data.index
y = data.values
ax.plot(x, y, 'o', label='$d_\mathrm{max}=60$')
fitx = np.linspace(x.min(), x.max(), 100)
fity = 0.692157 + 0.233396 * fitx
ax.plot(fitx, fity, '--', c=seaborn.xkcd_rgb["medium green"])

ax.set_xlabel('Number of flights in connected components')
ax.set_ylabel('Treewidth');
ax.legend()

pdf = matplotlib.backends.backend_pdf.PdfPages('../treewidth_connectivity.pdf');
pdf.savefig(figure=fig);
pdf.close();


# In[10]:

data = pd.read_csv('../data/graph_5.txt', sep='\t')
x = data.index
y = data.values
fig = plt.figure(figsize=fs)
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y, 'o-')
ax.set_xlabel('Treewidth')
ax.set_ylabel('PDF');

pdf = matplotlib.backends.backend_pdf.PdfPages('../treewidth_histogram.pdf');
pdf.savefig(figure=fig);
pdf.close();


# In[11]:

data = pd.read_csv('../data/graph_6.txt', sep='\t', header=None, skiprows=1).values
dmax = data[:, 0]
exponent = data[:, 1]
error = data[:, 2]
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(1, 1, 1)
ax.errorbar(dmax, exponent, yerr=error, fmt='o')
ax.set_xlabel('$d_\mathrm{max}$')
ax.set_ylabel('treewidth vs. connectivity slope');
pdf = matplotlib.backends.backend_pdf.PdfPages('../treewidth_pl.pdf');
pdf.savefig(figure=fig);
pdf.close();


# In[ ]:



