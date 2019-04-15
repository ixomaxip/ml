#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


zonds = pd.read_excel('data/пример данных.xlsx', sheet_name='зонды')
clusters = pd.read_excel('data/пример данных.xlsx', sheet_name='кластеры', skiprows=2)


# In[ ]:


zonds.info()
zonds.describe()
clusters.info()
clusters['p_code'] = clusters['poroda'].astype('category').cat.codes
# zonds.head()


# In[ ]:


zonds[zonds.well == 1]
poroda_group = clusters.groupby('p_code')

means = pd.DataFrame([poroda_group[fld].mean() for fld in ['q', 'f']])
std = np.transpose(pd.DataFrame([poroda_group[fld].std() for fld in ['q', 'f']]))


# In[ ]:


std[std['q'] != np.NaN]
clusters[clusters['p_code'] == 0]


# In[ ]:


cl_name = pd.DataFrame(pd.Series(pd.unique(clusters['poroda'])), columns=['poroda'])
idx = cl_name.keys()


# In[ ]:


cat = clusters.dtypes == object
cat_cols = clusters.columns[cat].tolist()

