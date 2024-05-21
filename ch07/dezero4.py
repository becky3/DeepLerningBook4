import matplotlib.pyplot as plt
import numpy as np
from dezero import Variable
import dezero.functions as F

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)


# 散布図のプロット
plt.scatter(x, y, s=10)
plt.show()
