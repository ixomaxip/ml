#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import sklearn as sk
from matplotlib import pyplot as plt
import seaborn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


zonds = pd.read_excel('data/пример данных.xlsx', sheet_name='зонды')
clusters = pd.read_excel('data/пример данных.xlsx', sheet_name='кластеры', skiprows=2)


# In[4]:


zonds.info()
zonds.describe()


# In[16]:


clusters['p_code'] = clusters['poroda'].astype('category').cat.codes
f_obj = ['well']
f_coords = ['X','Y','depth']
f_cat = ['poroda', 'p_code']
f_clust = clusters.columns.drop(f_cat).drop(f_coords).drop(f_obj)
f_zonds = zonds.columns.drop(f_obj).drop(f_coords)
wells = pd.unique(zonds['well'])
resp = ['poroda']
# zonds.head()


# In[ ]:





# In[22]:


np.array(f_clust.tolist() + resp)


# In[6]:


clusters[f_clust].describe()


# In[7]:


zonds[f_zonds].describe()


# In[ ]:





# In[8]:


zonds[zonds.well == 1]
poroda_group = clusters.groupby('p_code')

means = pd.DataFrame([poroda_group[fld].mean() for fld in ['q', 'f']])
std = np.transpose(pd.DataFrame([poroda_group[fld].std() for fld in ['q', 'f']]))


# In[9]:


std[std['q'] != np.NaN]
clusters[clusters['p_code'] == 0]


# In[10]:


cl_name = pd.DataFrame(pd.Series(pd.unique(clusters['poroda'])), columns=['poroda'])
idx = cl_name.keys()


# In[11]:


cat = clusters.dtypes == object
cat_cols = clusters.columns[cat].tolist()


# In[32]:


clusters[['q','f','p_code']].head()


# In[37]:


seaborn.pairplot(clusters[['X', 'Y', 'depth', 'q','f','poroda']], hue='poroda', diag_kind='kde')


# In[38]:


seaborn.pairplot(zonds, hue='well', diag_kind='kde')


# In[ ]:




