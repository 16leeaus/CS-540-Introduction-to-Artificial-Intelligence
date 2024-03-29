from collections import deque
import gym
import random
import numpy as np
import time
import pickle

from collections import defaultdict


EPISODES =   20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999



def default_Q_value():
    return 0


if __name__ == "__main__":




    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v0")
    env.seed(1)
    env.action_space.np_random.seed(1)


    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.

    episode_reward_record = deque(maxlen=100)


    for i in range(EPISODES):
        # Set episode reward to zero:
        episode_reward = 0
        # Create a new game enviornment:
        obs = env.reset()
        # Set done equal to false:
        done = False

        # While while game complete condtion is false:
        while done == False:

            if random.uniform(0,1) < EPSILON:
                action = env.action_space.sample()
            else:
                prediction = np.array([Q_table[(obs, i)] for i in range(env.action_space.n)])
                action = np.argmax(prediction)

            state = obs
            # Get parameters about the new enviornment:
            new, reward, done, info = env.step(action)

            # Set the new enviornment and add up the rewards:
            obs = new
            episode_reward += reward

            tempReward = max([Q_table[(obs, i)] for i in range(env.action_space.n)])

            # Update the Q-table accordingly:
            if not done:
                Q_table[state, action] += LEARNING_RATE * (reward + (DISCOUNT_FACTOR * tempReward) - Q_table[state, action])
            else:
                Q_table[state, action] += LEARNING_RATE * (reward - Q_table[state, action])
                episode_reward_record.append(episode_reward)

        EPSILON *= EPSILON_DECAY

        if i%100 ==0 and i>0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    ####DO NOT MODIFY######
    model_file = open('SARSA_Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    #######################



