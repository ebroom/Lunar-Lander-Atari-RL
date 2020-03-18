from Agent import Agent
from collections import deque
import gym
import numpy as np
import csv

episodes = 1600


def train(env, p):
    rewards = []
    last_100_rewards = deque([], maxlen=100)
    agent = Agent(p['e'], p['e_d'], p['mine'], p['g'], p['a'], p['maxm'], p['minm'], p['b'], env)
    for episode in range(episodes):
        total_reward = 0
        finished = False
        state = env.reset()
        while not finished:
            action = agent.get_action(state)
            previous_state = state
            state, reward, finished, info = env.step(action)
            #env.render()
            agent.save_memory(previous_state, action, state, reward, finished)
            agent.learn()
            total_reward += reward
        agent.update_target_nn()
        agent.update_epsilon()
        rewards.append(total_reward)
        last_100_rewards.append(total_reward)
        if episode % 20 == 0:
            print("Episode " + str(episode))
            print("Mean Reward: " + str(np.mean(last_100_rewards)))
        if np.mean(last_100_rewards) > 200:
            print("We won!")
            break
    with open('lunarlander_alpha_' + str(p['a']) + "_gamma_" + str(p['g']) + "_epislon_" + str(p['mine']) + '.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(rewards)
    return agent, np.mean(last_100_rewards)


def test(env, agent, p):
    rewards = []
    for episode in range(100):
        state = env.reset()
        total_reward = 0
        finished = False
        while not finished:
            action = agent.get_action(state)
            state, reward, finished, info = env.step(action)
            #env.render()
            total_reward += reward
        rewards.append(total_reward)
    with open('lunarlander_test_alpha_' + str(p['a']) + "_gamma_" + str(p['g']) + "_epislon_" + str(p['mine']) + '.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(rewards)


def run():
    env = gym.make('LunarLander-v2')
    # Epsilon, Epsilon_decay, gamma, alpha, memory size, min memory, batch size
    parameters = [
        {'e': 1.0, 'e_d': .998, 'mine': 0.1, 'g': .999, 'a': .0001, 'maxm': 65536, 'minm': 64, 'b': 32}
    ]
    for p in parameters:
        agent, reward = train(env, p)
        if reward > 200:
            test(env, agent, p)


if __name__ == '__main__':
    run()
