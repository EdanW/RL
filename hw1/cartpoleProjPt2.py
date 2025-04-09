import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from cartpoleProjPt1 import *

def rand_srch():
    sample_times = 10000
    episode_counter = 1
    env = gym.make('CartPole-v1')
    while episode_counter <= sample_times:
        weights = np.random.uniform(-1,1,size=4) #inits
        curr_rwrd = episode(env, weights)
        if (curr_rwrd>=200):
            break
        episode_counter += 1
    env.close()
    return episode_counter

if __name__ == '__main__':
    random_search_cases = 1000
    episodes_required_list = []
    for i in range(random_search_cases):
        print(i)
        episodes_required_list.append(rand_srch())

    print(f'avg episodes: {np.mean(episodes_required_list)}')
    plt.hist(episodes_required_list, bins = 50)
    plt.xlabel("episodes required for 200 reward")
    plt.ylabel("amount", fontweight='bold')
    plt.annotate(f'The average number of episodes required is {np.mean(episodes_required_list)}', xy = (0.2, 0.7), xycoords='axes fraction')
    plt.show()
