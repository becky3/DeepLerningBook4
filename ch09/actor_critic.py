import matplotlib.pyplot as plt
import numpy as np
import gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L


class PolicyNet(Model):
    def __init__(self, action_size=2):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = F.softmax(x)
        return x


class ValueNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.0005
        self.action_size = 2

        self.pi = PolicyNet(self.action_size)
        self.v = ValueNet()

        self.optimizer_pi = optimizers.Adam(self.lr_pi).setup(self.pi)
        self.optimizer_v = optimizers.Adam(self.lr_v).setup(self.v)

    def get_action(self, state):
        state = state[np.newaxis, :]
        probs = self.pi(state)
        probs = probs[0]
        action = np.random.choice(len(probs), p=probs.data)
        return action, probs[action]

    def update(self, state, action_prob, reward, next_state, done):
        # バッチ軸の追加
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]

        # (1) self.v の損失
        target = reward + self.gamma * self.v(next_state) * (1 - done)
        target.unchain()
        v = self.v(state)
        loss_v = F.mean_squared_error(v, target)

        # (2) self.pi の損失
        delta = target - v
        delta.unchain()
        loss_pi = -F.log(action_prob) * delta

        self.v.cleargrads()
        self.pi.cleargrads()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.update()
        self.optimizer_pi.update()


episodes = 1000
env = gym.make("CartPole-v1", render_mode="human")

agent = Agent()
reward_history = []

for episode in range(episodes):
    state = env.reset()
    state = state[0] if isinstance(state, tuple) else state

    done = False
    total_reward = 0

    while not done:
        action, prob = agent.get_action(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = next_state[0] if isinstance(next_state, tuple) else next_state

        agent.update(state, prob, reward, next_state, done)

        state = next_state
        total_reward += reward

    reward_history.append(total_reward)

    print(f"[{episode}]{total_reward}")


reward_history_np = np.array(reward_history)
reward_means = np.array(
    [
        reward_history_np[i : i + 100].mean()
        for i in range(0, len(reward_history_np), 100)
    ]
)

plt.plot(reward_history, label="Reward per episode")
plt.plot(
    np.arange(0, len(reward_history), 100),
    reward_means,
    label="Average reward per 100 episodes",
)
plt.title("Reward History")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.show()
