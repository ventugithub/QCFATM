
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import seaborn
import pandas as pd
#get_ipython().magic(u'matplotlib inline')


# In[3]:

pltData = pd.read_hdf('preprocessing_data.h5', 'data')


# In[4]:

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
pltData.plot(ax=ax)
ax.set_yscale('log')
ax.set_ylabel('Number of conflicts')
ax.set_xlabel('$D_{max}$ / min')
pdf = matplotlib.backends.backend_pdf.PdfPages('preprocessing_reduction_number_of_conflicts.pdf');
pdf.savefig(figure=fig);
pdf.close();


# In[ ]:



