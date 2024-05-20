import numpy as np


def greedy_probs(Q, state, epsilon=0, action_size=4):
    qa = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qa)

    base_probs = epsilon / action_size
    action_probs = {action: base_probs for action in range(action_size)}
    action_probs[int(max_action)] += 1 - epsilon
    return action_probs
