from collections import defaultdict

import numpy as np

from common.GridWorld import GridWorld


class TdAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.1
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.V = defaultdict(lambda: 0)

    def get_action(self, state):
        action_prob = self.pi[state]
        actions = list(action_prob.keys())
        probs = list(action_prob.values())
        return np.random.choice(actions, p=probs)

    def eval(self, state, reward, next_state, done):
        next_v = 0 if done else self.V[next_state]  # ゴールの価値関数は0
        target = reward + self.gamma * next_v

        self.V[state] += (target - self.V[state]) * self.alpha


env = GridWorld()
agent = TdAgent()

episodes = 1000
for episode in range(episodes):

    state = env.reset()
    done = False

    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.eval(state, reward, next_state, done)  # 毎回呼ぶ
        if done:
            break

        state = next_state

env.render_v(agent.V)
