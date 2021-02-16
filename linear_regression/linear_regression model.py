import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('D:\desktop\py_csv\dataset1.CSV')
xy = np.array(data.iloc[:, [0, 1]])
z = np.array((data.iloc[:, [2]]))
x = np.array(data.iloc[:, [0]])
y = np.array(data.iloc[:, [1]])

lrg = LinearRegression()
lrg.fit(xy, z)
score = lrg.score(xy, z)
print('coefficients=', lrg.coef_, ' ', 'intercept=', lrg.intercept_, '\n')
print('score=', score, '\n')

fig = plt.figure(figsize=(5, 5))
ax = Axes3D(fig)
ax.scatter(x, y, z, label='first curve')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title("linear regression")
plt.show()
