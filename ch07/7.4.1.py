import numpy as np


def one_hot(state):
    HEIGHT, WIDTH = 3, 8
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)
    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0
    return vec[np.newaxis, :]


state = (2, 0)
x = one_hot(state)

print(x.shape)
print(x)
