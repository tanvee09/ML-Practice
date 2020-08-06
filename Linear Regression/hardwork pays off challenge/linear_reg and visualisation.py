#!/usr/bin/env python
# coding: utf-8

# - Download
# - Load
# - Visualise
# - Normalisation

# In[1]:


from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


dfx = pd.read_csv('./train/Linear_X_Train.csv')
dfy = pd.read_csv('./train/Linear_Y_Train.csv')

print(dfx.shape)
print(dfy.shape)


# In[3]:


dfx.head(n = 10)


# In[4]:


dfy.head()


# In[5]:


X = dfx.values
y = dfy.values


# In[6]:


print(X.shape)
print(y.shape)

print(X)
print(y)


# ## Visualisation

# In[7]:


plt.style.use("seaborn")
plt.scatter(X, y)
plt.title("Time Spent vs Performance Graph")
plt.xlabel("Time Spent")
plt.ylabel("Performance")
plt.show()


# Clearly, the problem can be solved using linear regression.

# ## Normalisation

# In[8]:


u = X.mean()
std = X.std()


# In[9]:


print(u)
print(std)


# Since std is close to 1, data is already normalised.

# In[10]:


X = (X - u) / std


# In[11]:


X


# # Linear Regression

# Only for single feature problems

# In[12]:


def hypothesis(x, theta) :
    # theta = [theta0, theta1]
    y_ = theta[0] + theta[1] * x
    return y_


# In[13]:


def gradient(X, Y, theta) :
    m = X.shape[0]
    grad = np.zeros((2,))
    for i in range(m) :
        x = X[i]
        y_ = hypothesis(x, theta)
        y = Y[i]
        grad[0] += (y_ - y)
        grad[1] += (y_ - y) * x
    return grad / m


# In[14]:


def error(X, Y, theta) :
    m = X.shape[0]
    total_error = 0.0
    for i in range(m) :
        x = X[i]
        y_ = hypothesis(x, theta)
        y = Y[i]
        total_error += (y_ - y) ** 2
    return total_error / (m)


# In[15]:


def gradientDescent(X, Y, max_steps = 100,learning_rate = 0.1) :
    theta = np.zeros((2, ))
    error_list = []
    theta_list = []
    for i in range(max_steps) :
        x = X[i]
        grad = gradient(X, Y, theta)
        e = error(X, Y, theta)
        error_list.append(e)
        theta[0] = theta[0] - learning_rate * grad[0]
        theta[1] = theta[1] - learning_rate * grad[1]
        theta_list.append((theta[0], theta[1]))
    return theta, error_list, theta_list


# In[16]:


theta, error_list, theta_list = gradientDescent(X, y)


# In[17]:


print(theta)


# In[18]:


error_list


# In[19]:


plt.plot(error_list)
plt.title("Reduction in error over time")
plt.show()


# ## Predcitions and Best Line

# y_ = theta[0] + theta[1] * X

# In[20]:


y_ = hypothesis(X, theta)


# In[21]:


plt.scatter(X, y)
plt.plot(X, y_, color = 'orange', label = 'prediction')
plt.legend()
plt.show()


# In[22]:


df_test = pd.read_csv('./test/Linear_X_Test.csv')
print(df_test.shape)


# In[23]:


X_test = df_test.values


# In[24]:


Y_test = hypothesis(X_test, theta)


# In[25]:


print(Y_test)


# In[26]:


plt.scatter(X_test, Y_test, label = 'predictions')
plt.plot(X, y_, color = 'orange', label = 'base line')
plt.legend()
plt.show()


# In[27]:


test_df = pd.DataFrame(data = Y_test, columns = ['y'])


# In[28]:


test_df.head()


# In[29]:


test_df.to_csv('y_prediction.csv', index = False)


# ## Computing Score (on training data)
# **Score:** R2 (R-squared) or Coefficient of Determination

# In[30]:


def r2_score(Y, y_) :
    num = np.sum((Y - y_)**2)
    denom = np.sum((Y - Y.mean())**2)
    score = 1 - (num / denom)
    return score * 100


# In[31]:


print(r2_score(y, y_))


# ## Visualising Loss Function, Gradient Descent, Theta Updates

# In[32]:


theta


# In[33]:


T0 = np.arange(-50, 50, 1)
T1 = np.arange(40, 120, 1)

print(T0)
print(T1)


# In[34]:


t0, t1 = np.meshgrid(T0, T1)
print(t0.shape)


# In[44]:


J = np.zeros(t0.shape)

for i in range(J.shape[0]) :
    for j in range(J.shape[1]) :
        y_ = t1[i, j] * X + t0[i, j]
        J[i, j] = np.sum((y - y_) ** 2) / y.shape[0]
    


# In[45]:


fig = plt.figure()
axes = plt.gca(projection = '3d')
axes.plot_surface(t0, t1, J, cmap = 'rainbow')
plt.show()


# In[46]:


fig = plt.figure()
axes = plt.gca(projection = '3d')
axes.contour(t0, t1, J, cmap = 'rainbow')
plt.show()


# In[47]:


theta_list[:5]


# In[48]:


# Plot changes in values of theta


# In[49]:


theta_list = np.array(theta_list)
theta_list[:5]


# In[50]:


plt.plot(theta_list[:, 0], label = 'theta 0')
plt.plot(theta_list[:, 1], label = 'theta 1')
plt.legend()
plt.show()


# In[51]:


# Trajectory traced by theta updates in the loss function


# In[53]:


fig = plt.figure()
axes = plt.gca(projection = '3d')
axes.plot_surface(t0, t1, J, cmap = 'rainbow')
axes.scatter(theta_list[:, 0], theta_list[:, 1], error_list, color = 'black')
plt.show()


# In[54]:


fig = plt.figure()
axes = plt.gca(projection = '3d')
axes.contour(t0, t1, J, cmap = 'rainbow')
axes.scatter(theta_list[:, 0], theta_list[:, 1], error_list, color = 'black')
plt.show()


# ## 2D Contour Plot

# In[57]:


plt.contour(t0, t1, J, cmap = 'rainbow')
plt.scatter(theta_list[:, 0], theta_list[:, 1], color = 'black')
plt.show()


# In[ ]:




