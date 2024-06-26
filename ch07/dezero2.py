﻿import numpy as np
from dezero import Variable


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0**2) ** 2 + (1 - x0) ** 2
    return y


xt0 = Variable(np.array(0.0))
xt1 = Variable(np.array(2.0))


y = rosenbrock(xt0, xt1)
y.backward()
print(xt0.grad, xt1.grad)

print("===")

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

lr = 0.001  # 学習率
iters = 10000  # 繰り返し回数

for i in range(iters):
    print(x0, x1)
    y = rosenbrock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    x0.data -= lr * x0.grad.data
    x1.data -= lr * x1.grad.data

print(x0, x1)
