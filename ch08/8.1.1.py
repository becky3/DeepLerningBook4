import gym

env = gym.make("CartPole-v0")

state = env.reset()
print(state)

action_space = env.action_space
print(action_space)

action = 0
next_state, reward, terminated, truncated, info = env.step(action)
print(next_state)
