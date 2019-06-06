#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import sklearn as sk
from matplotlib import pyplot as plt
import seaborn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


zonds = pd.read_excel('data/пример данных.xlsx', sheet_name='зонды')
clusters = pd.read_excel('data/пример данных.xlsx', sheet_name='кластеры', skiprows=2)
zonds.shape


# In[15]:


zonds.dtypes


# In[30]:


zond = pd.read_excel('data/data.xlsx', sheet_name='ZOND')
zond.dtypes


# In[34]:


zond['Q'] = zond['Q'].apply(pd.to_numeric, errors='coerce')
zond['F'] = zond['F'].apply(pd.to_numeric, errors='coerce')


# In[35]:


zond.describe()


# In[36]:


zond.dtypes


# In[5]:


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





# In[6]:


np.array(f_clust.tolist() + resp)


# In[7]:


clusters[f_clust].describe()


# In[8]:


zonds[f_zonds].describe()


# In[ ]:





# In[9]:


zonds[zonds.well == 1]
poroda_group = clusters.groupby('p_code')

means = pd.DataFrame([poroda_group[fld].mean() for fld in ['q', 'f']])
std = np.transpose(pd.DataFrame([poroda_group[fld].std() for fld in ['q', 'f']]))


# In[10]:


std[std['q'] != np.NaN]
clusters[clusters['p_code'] == 0]


# In[11]:


cl_name = pd.DataFrame(pd.Series(pd.unique(clusters['poroda'])), columns=['poroda'])
idx = cl_name.keys()


# In[12]:


cat = clusters.dtypes == object
cat_cols = clusters.columns[cat].tolist()


# In[13]:


clusters[['q','f','p_code']].head()


# In[37]:


seaborn.pairplot(clusters[['X', 'Y', 'depth', 'q','f','poroda']], hue='poroda', diag_kind='kde')


# In[58]:


seaborn.pairplot(zonds, hue='well', diag_kind='kde')


# In[39]:


seaborn.heatmap(clusters[f_clust].corr(), square=True)


# In[41]:


zonds.head()


# In[43]:


seaborn.heatmap(zonds[['depth', 'q', 'f']].corr(), square=True)


# In[45]:


wells[0]


# # Clustering

# In[1]:


from sklearn.model_selection import train_test_split


# In[27]:


clusters['p_code'].hist(bins=len(pd.unique(clusters['poroda'])), alpha=0.5)
data = clusters[]


# In[37]:


data = clusters[clusters.columns.drop(f_cat).drop(f_obj).drop(f_coords)]


# ## Scaling

# In[86]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)


# ## PCA

# In[88]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# ## Plot

# In[89]:


def discrete_scatter(x1, x2, y=None, markers=None, s=10, ax=None,
                     labels=None, padding=.2, alpha=1, c=None, markeredgewidth=None):
    import matplotlib as mpl
    from matplotlib.colors import ListedColormap, colorConverter, LinearSegmentedColormap
    if ax is None:
        ax = plt.gca()

    if y is None:
        y = np.zeros(len(x1))

    unique_y = np.unique(y)

    if markers is None:
        markers = ['o', '^', 'v', 'D', 's', '*', 'p', 'h', 'H', '8', '<', '>'] * 10

    if len(markers) == 1:
        markers = markers * len(unique_y)

    if labels is None:
        labels = unique_y

    # lines in the matplotlib sense, not actual lines
    lines = []

    current_cycler = mpl.rcParams['axes.prop_cycle']

    for i, (yy, cycle) in enumerate(zip(unique_y, current_cycler())):
        mask = y == yy
        # if c is none, use color cycle
        if c is None:
            color = cycle['color']
        elif len(c) > 1:
            color = c[i]
        else:
            color = c
        # use light edge for dark markers
        if np.mean(colorConverter.to_rgb(color)) < .4:
            markeredgecolor = "grey"
        else:
            markeredgecolor = "black"

        lines.append(ax.plot(x1[mask], x2[mask], markers[i], markersize=s,
                             label=labels[i], alpha=alpha, c=color,
                             markeredgewidth=markeredgewidth,
                             markeredgecolor=markeredgecolor)[0])

    if padding != 0:
        pad1 = x1.std() * padding
        pad2 = x2.std() * padding
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(min(x1.min() - pad1, xlim[0]), max(x1.max() + pad1, xlim[1]))
        ax.set_ylim(min(x2.min() - pad2, ylim[0]), max(x2.max() + pad2, ylim[1]))

    return lines


# In[90]:


plt.figure(figsize=(20,20))
discrete_scatter(X_pca[:,0], X_pca[:,1],clusters['p_code']);
plt.legend(clusters['poroda'], loc='best', bbox_to_anchor=(1.5, -0.05), ncol=3)
plt.gca().set_aspect('equal')


# ## Robust

# In[91]:


from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaledR = scaler.fit_transform(data)


# In[92]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pcaR = pca.fit_transform(X_scaledR)


# In[95]:


plt.figure(figsize=(15,15))
discrete_scatter(X_pcaR[:,0], X_pcaR[:,1],clusters['poroda']);
plt.legend(clusters['poroda'], loc='best', bbox_to_anchor=(1.5, -0.05), ncol=3)
plt.gca().set_aspect('equal')


# In[ ]:




