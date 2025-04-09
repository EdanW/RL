import gymnasium as gym
import numpy as np

def act(obs, weights):
    if np.matmul(obs, weights) >= 0:
        return 1 # go right
    return 0 # go left according to action space

def episode(env, weights):
    acc_rewards = 0
    obs, info = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = act(weights, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        acc_rewards += reward
    return acc_rewards
        

if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode="human")
    weights = np.random.uniform(-1,1,size=4) #inits
    score=episode(env, weights)
    print(f'episode score {score}')

