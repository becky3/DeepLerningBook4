from common.GridWorld import GridWorld
from collections import defaultdict


def eval_onestep(pi, V, env, gamma=0.9):
    for state in env.states():  # (1) 各状態へアクセス
        if state == env.goal_state:  # (2) ゴールの価値観数は常に0
            V[state] = 0
            continue

        action_probs = pi[state]  # probs は probabilities の略
        new_V = 0

        # (3) 各行動へアクセス
        for action, action_probs in action_probs.items():
            next_state = env.next_step(state, action)
            r = env.reward(state, action, next_state)
            # (4) 新しい価値関数
            new_V += action_probs * (r + gamma * V[next_state])
        V[state] = new_V

    return V


def policy_eval(pi, V, env, gamma=0.9, threshold=0.001):
    while True:
        old_V = V.copy()  # 更新前の価値関数
        V = eval_onestep(pi, V, env, gamma)

        # 更新された量の最大値を求める
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        # 閾値との比較
        if delta < threshold:
            break

    return V


env = GridWorld()
gamma = 0.9

pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
V = defaultdict(lambda: 0.0)

V = policy_eval(pi, V, env, gamma)
env.render_v(V, pi)
