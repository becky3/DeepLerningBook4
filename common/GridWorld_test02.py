from collections import defaultdict
from common.GridWorld import GridWorld


env = GridWorld()
V = defaultdict(lambda: 0.0)

state = (1, 2)
print(V[state])

print("===")

pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})

state = (0, 1)
print(pi[state])
