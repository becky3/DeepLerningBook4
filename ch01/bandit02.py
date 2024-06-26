﻿import os
import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, arms=10):
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0


class Agent:
    def __init__(self, epsilon, actions=10):
        self.epsilon = epsilon
        self.ns = np.zeros(actions)
        self.Qs = np.zeros(actions)

    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        else:
            return np.argmax(self.Qs)


steps = 1000
epsilon = 0.1

bandit = Bandit()
agent = Agent(epsilon)

total_reward = 0
total_rewards = []
rates = []

for step in range(steps):
    action = agent.get_action()
    reward = bandit.play(action)
    agent.update(action, reward)
    total_reward += reward

    total_rewards.append(total_reward)
    rates.append(total_reward / (step + 1))

print("Total Reward:", total_reward)

path = os.path.dirname(os.path.abspath(__file__))

print(path)

plt.ylabel("Total Reward")
plt.xlabel("Steps")
plt.plot(total_rewards)
plt.show()

# plt.savefig(os.path.join(path, "bandit02_total_rewards.png"))

plt.ylabel("Rates")
plt.xlabel("Steps")
plt.plot(rates)
plt.show()

# plt.savefig(os.path.join(path, "bandit02_rates.png"))
