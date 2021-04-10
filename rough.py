import tensorflow as tf
import gym
env = gym.make('BipedalWalker-v3')
obs = env.reset()

for i in range(100):
    obs, reward, info, done = env.step(env.action_space.sample())
    env.render()
    print(env.action_space.sample())
