import numpy as np


def sample(dices=2):
    x = 0
    for _ in range(dices):
        x += np.random.choice(range(1, 7))
    return x


print(sample())
print(sample())
print(sample())
