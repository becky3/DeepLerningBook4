from common.GridWorld import GridWorld

env = GridWorld()
action = 0
next_state, reward, done = env.step(action)

print("next_state:", next_state)
print("reward:", reward)
print("done:", done)
