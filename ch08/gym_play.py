import numpy as np
import gym

env = gym.make("CartPole-v1", render_mode="human")
state = env.reset()
terminated = False

while not terminated:
    env.render()
    action = np.random.choice([0, 1])
    next_state, reward, terminated, truncated, info = env.step(action)
    print(f"{action}, {terminated}")

env.close()
