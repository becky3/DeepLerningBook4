import numpy as np

x = np.array([1, 2, 3])
pi = np.array([0.1, 0.1, 0.8])

# 期待値
e = np.sum(x * pi)
print("E_pi[x] =", e)

# モンテカルロ法
n = 100
samples = []
for _ in range(n):
    samples.append(np.random.choice(x, p=pi))  # piに従ってxをサンプリング

mean = np.mean(samples)
var = np.var(samples)
print("MC: {:.2f} (var: {:.2f})".format(mean, var))
