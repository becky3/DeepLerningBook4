import matplotlib.pyplot as plt
import numpy as np
from dezero import Variable
import dezero.functions as F

# トイデータセット
np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))


def predict(x):
    y = F.matmul(x, W) + b
    return y


def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff**2) / diff.size


lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data

    if i % 10 == 0:
        print(loss.data)

print("====")
print("W =", W.data)
print("b =", b.data)


# プロットの追加
plt.scatter(x.data, y.data, s=10)  # 散布図のプロット
plt.plot(x.data, predict(x).data, color="red")  # 予測値のプロット
plt.show()  # プロットの表示
