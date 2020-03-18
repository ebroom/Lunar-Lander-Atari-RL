from collections import deque
import numpy as np
import random
from NN import NN


class Agent:

    def __init__(self, epsilon, epsilon_decay, min_epsilon, gamma, alpha, max_memory, min_memory, batch_size, env):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.env = env
        self.max_memory = max_memory
        self.min_memory = min_memory
        self.memory = deque([], self.max_memory)
        self.nn = NN(alpha)
        self.target_nn = NN(alpha)
        self.batch_size = batch_size

    def save_memory(self, previous_state, action, state, reward, finished):
        self.memory.append({'previous_state': previous_state, 'action': action, 'state': state, 'reward': reward, 'finished': finished})

    def get_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            state_array = np.asarray(state).reshape((1, 8))
            return np.argmax(self.nn.predict(state_array))

    def get_samples_from_batch(self):
        samples = random.sample(self.memory, self.batch_size)
        previous_states = []
        actions = []
        states = []
        rewards = []
        finishes = []

        for sample in samples:
            previous_states.append(sample['previous_state'])
            actions.append(sample['action'])
            states.append(sample['state'])
            rewards.append(sample['reward'])
            finishes.append(sample['finished'])

        previous_states = np.asarray(previous_states)
        actions = np.asarray(actions)
        states = np.asarray(states)
        rewards = np.asarray(rewards)
        finishes = np.asarray(finishes)
        return self.q_learning(len(samples), previous_states, actions, states, rewards, finishes)

    def q_learning(self, length, previous_states, actions, states, rewards, finishes):
        Q = self.nn.predict(previous_states)
        Q_target = self.target_nn.predict(states)
        y = np.zeros((length, 4))
        for i in range(length):
            reward, action, Q_cur = rewards[i], actions[i], Q[i]
            if finishes[i]:
                Q_cur[action] = reward
            else:
                Q_cur[action] = reward + self.gamma * np.max(Q_target[i])

            y[i] = Q_cur

        return previous_states, y

    def learn(self):
        if len(self.memory) > self.min_memory and len(self.memory) > self.batch_size:
            x, y = self.get_samples_from_batch()
            self.nn.train(x, y, batch_size=self.batch_size)

    def update_target_nn(self):
        self.target_nn.set_weights(self.nn.get_weights())

    def update_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon * self.epsilon_decay
