#!/usr/bin/env python
# coding: utf-8

# In[32]:


from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib notebook
# for interactibe plotting


# In[33]:


X, y = make_regression()


# In[34]:


print(X.shape, y.shape)


# In[35]:


X, y = make_regression(n_samples = 1000, n_features = 2, n_informative = 1, noise = 10, random_state = 1)
# random_state helps in getting same data every time, n_informative is no. of informative features (not useful features)


# In[36]:


print(X.shape, y.shape)


# In[37]:


plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], y)
plt.subplot(1, 2, 2)
plt.scatter(X[:, 1], y)
plt.show()


# In[40]:


from mpl_toolkits import mplot3d

fig = plt.figure(figsize = (5, 5))
ax = plt.axes(projection = '3d')

ax.scatter3D(X[:, 0], X[:, 1], y, color = 'green');
plt.title('3D scatter plot')
plt.show()


# In[ ]:




