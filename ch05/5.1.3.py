import numpy as np


def sample(dices=2):
    x = 0
    for _ in range(dices):
        x += np.random.choice(range(1, 7))
    return x


trial = 1000

samples = []

for i in range(trial):
    s = sample()
    samples.append(s)
    V = sum(samples) / len(samples)  # 毎回、平均値を計算する
    print(V)
