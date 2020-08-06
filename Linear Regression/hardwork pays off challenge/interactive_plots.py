import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

X = pd.read_csv('./train/Linear_X_Train.csv').values
Y = pd.read_csv('./train/Linear_Y_Train.csv').values

theta = np.load('ThetaList.npy')

# We'll visualise how the base line looks after every iteration

T0 = theta[:, 0]
T1 = theta[:, 1]

plt.ion() # Interactive on
for i in range(0, 60, 2) :
    y_ = T1[i] * X + T0
    plt.scatter(X, Y)
    plt.plot(X, y_, 'red')
    plt.draw()
    plt.pause(0.05) # Pause for one second
    plt.clf() # Clear last object