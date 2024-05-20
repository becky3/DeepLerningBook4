import numpy as np

x = np.array([1, 2, 3])
pi = np.array([0.1, 0.1, 0.8])

b = np.array([0.2, 0.2, 0.6])
n = 100
samples = []

for _ in range(n):
    idx = np.arange(len(b))
    i = np.random.choice(idx, p=b)  # bを使ってサンプリング
    s = x[i]
    rho = pi[i] / b[i]
    samples.append(s * rho)

mean = np.mean(samples)
var = np.var(samples)
print("IS: {:.2f} (var: {:.2f})".format(mean, var))
