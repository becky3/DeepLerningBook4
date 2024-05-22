import numpy as np
from dezero import Model
import dezero.layers as L
import dezero.functions as F


def one_hot(state):
    HEIGHT, WIDTH = 3, 8
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)
    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0
    return vec[np.newaxis, :]


class QNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(100)
        self.l2 = L.Linear(4)

    def forward(self, x):
        h = F.sigmoid(self.l1(x))
        h = self.l2(h)
        return h


qnet = QNet()

state = (2, 0)
state = one_hot(state)

qs = qnet(state)
print(qs.shape)
