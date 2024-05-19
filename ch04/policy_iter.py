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


def argmax(d):
    max_value = max(d.values())
    max_key = 0
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key


def greedy_policy(V, env, gamma):
    pi = {}

    for state in env.states():
        action_values = {}

        for action in env.actions():
            next_state = env.next_step(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values[action] = value

        max_action = argmax(action_values)
        action_probs = {0: 0, 1: 0, 2: 0, 3: 0}
        action_probs[max_action] = 1.0
        pi[state] = action_probs
    return pi


def policy_iter(env, gamma=0.9, threshold=0.001, is_render=False):
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0.0)

    while True:
        V = policy_eval(pi, V, env, gamma, threshold)
        new_pi = greedy_policy(V, env, gamma)

        if is_render:
            env.render_v(V, pi)

        if pi == new_pi:
            break

        pi = new_pi

    return pi


def value_iter_onestep(V, env, gamma):
    for state in env.states():  # (1) すべての状態へアクセス
        if state == env.goal_state:  # ゴールの価値関数は常に0
            V[state] = 0
            continue

        action_values = []

        for action in env.actions():  # (2) すべての行動にアクセス
            next_state = env.next_step(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]  # (3) 新しい価値関数
            action_values.append(value)

        V[state] = max(action_values)  # (4) 最大値を取り出す

    return V


def value_iter(V, env, gamma=0.9, threshold=0.001, is_render=True):
    while True:
        if is_render:
            env.render_v(V)

        old_V = V.copy()  # 更新前の価値関数
        V = value_iter_onestep(V, env, gamma)

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


# action_values = {0: 0.1, 1: -0.3, 2: 9.9, 3: -1.3}
#
# max_action = argmax(action_values)
# print(max_action)
#
# print("===")
#
# env = GridWorld()
# gamma = 0.9
# pi = policy_iter(env, gamma, is_render=True)

V = defaultdict(lambda: 0.0)
env = GridWorld()
gamma = 0.9

V = value_iter(V, env, gamma)

pi = greedy_policy(V, env, gamma)
env.render_v(V, pi)
