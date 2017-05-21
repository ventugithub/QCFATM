
# coding: utf-8

# In[6]:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import seaborn
#get_ipython().magic(u'matplotlib inline')


# In[10]:

infile = open('connected_components.npz', 'r')
npzfile = np.load(infile)
nc = npzfile["nc"]
nctrivial = npzfile["nctrivial"]
infile.close()


# In[11]:

delays = [6, 9, 12, 15, 18, 24, 36, 48, 60]

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(1, 1, 1)

ax.plot(delays, np.array(nc) + np.array(nctrivial), 'o-', label='including trivial connected components')
ax.plot(delays, nc, 'o-', label='excluding trivial connected components')

ax.set_ylabel('number of connected components')
ax.set_xlabel('$d_\mathrm{max}$')
ax.legend()

pdf = matplotlib.backends.backend_pdf.PdfPages('num_cc.pdf');
pdf.savefig(figure=fig);
pdf.close();


# In[ ]:



