from common.GridWorld import GridWorld
import numpy as np

env = GridWorld()

print(env.hight)
print(env.width)
print(env.shape)

print("===")

for action in env.actions():
    print(action)

print("===")

for state in env.states():
    print(state)

print("===")


env = GridWorld()
env.render_v()


env = GridWorld()
V = {}
for state in env.states():
    V[state] = np.random.rand()
env.render_v(V)
