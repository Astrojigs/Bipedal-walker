import numpy as np
import tensorflow as tf
import gym
import random
from collections import deque



class DeepQNetwork:
    def __init__(self,env, epsilon, gamma, lr):
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma
        self.counter = 0

        # Buffer
        self.replay_buffer = deque(maxlen=500000)
        self.batch_size = 64
        self.rewards_list = []
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        # Environment
        self.env = env
        # Action space
        self.action_space = self.env.action_space
        self.num_action_space = self.action_space.sample().size # = 4
        # Observation space
        self.observation_space = self.env.observation_space
        self.num_observation_space = self.observation_space.shape[0] # = 24

        # Model
        self.model = self.initialize_model()

    def initialize_model(self):
        one = tf.keras.layers.Input(shape=[self.num_observation_space])
        input_layer = tf.keras.layers.Dense(256, activation=tf.keras.activations.elu)(one)
        second_layer = tf.keras.layers.Dense(512, activation='elu')(input_layer)
        middle_layer = tf.keras.layers.Dense(256, activation=tf.keras.activations.elu)(second_layer)
        last_layer = tf.keras.layers.Dense(self.num_action_space, activation='linear')(middle_layer)
        model = tf.keras.Model(inputs=[one], outputs=[last_layer])
        model.compile(tf.keras.optimizers.Adam(lr=self.lr),loss=tf.keras.losses.mean_squared_logarithmic_error)
        return model

    def get_action(self, state):
        # Epsilon Greedy policy
        if np.random.rand() > self.epsilon:
            print(f"Action received from the model is: {self.model.predict(state)[0]}")
            return np.clip(self.model.predict(state)[0], -1, 1)


        else:
            return self.env.action_space.sample()

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update_counter(self):
        self.counter+=1
        step_size = 5
        self.counter = self.counter % step_size

    def get_attributes_from_sample(self, sample):
        states = np.squeeze(np.squeeze(np.array([i[0] for i in sample])))
        actions = np.array([i[1] for i in sample])
        rewards = np.array([i[2] for i in sample])
        next_states = np.squeeze(np.array([i[3] for i in sample]))
        done_list = np.array([i[4] for i in sample])
        return states, actions, rewards, next_states, done_list


    def update_model(self):
        # replay_buffer size Check
        if len(self.replay_buffer) < self.batch_size or self.counter != 0:
            return

        # Early Stopping
        if np.mean(self.rewards_list[-10:]) > 180:
            return

        # take a random sample:
        random_sample = random.sample(self.replay_buffer, self.batch_size)
        # Extract the attributes from sample
        states, actions, rewards, next_states, done_list = self.get_attributes_from_sample(random_sample)
        targets = np.tile(rewards, (self.num_action_space, 1)).T + np.multiply(np.tile((1 - done_list), (self.action_space.sample().size, 1)).T, np.multiply(self.gamma, self.model.predict_on_batch(next_states)))
        # print(targets.shape) = (64,)
        target_vec = self.model.predict_on_batch(states) # shape = (64, 4)
        indexes = np.array([i for i in range(self.batch_size)])
        target_vec = targets

        self.model.fit(states, target_vec, epochs=1, verbose=0)

    def learn(self, num_episodes = 2000):
        for episode in range(num_episodes):
            #reset the environment
            state = self.env.reset()

            reward_for_episode = 0
            num_steps = 1600
            state = np.reshape(state, [1,self.num_observation_space])
            #what to do in every step
            for step in range(num_steps):
                # Get the action
                received_action = self.get_action(state)
                # print(f"From learn() || received_action = {received_action} and it's shape = {received_action.shape}")
                # From learn() || received_action = 1 and it's shape = ()

                # Implement the action and the the next_states and rewards
                next_state, reward, done, info = env.step(received_action)

                # Render the actions
                self.env.render()

                # Reshape the next_state and put it in replay buffer
                next_state = np.reshape(next_state, [1, self.num_observation_space])
                # Store the experience in replay_buffer
                self.add_to_replay_buffer(state, received_action, reward, next_state, done)

                # Add rewards
                reward_for_episode+=reward
                # Change the state
                state = next_state

                # Update the model
                self.update_counter()
                self.update_model()

                if done:
                    break

            self.rewards_list.append(reward_for_episode)

            # Decay the epsilon after each completion
            if self.epsilon < 0.6:
                if np.mean(self.rewards_list[-100:])>np.mean(self.rewards_list[-200:]):
                    if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.epsilon_decay
            else:
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

            last_reward_mean = np.mean(self.rewards_list[-100:])
            if last_reward_mean > 200:
                print("DQN Training Complete...")
                break

            # Saving the Model
            # self.model.save('LL1_model.h5', overwrite=True)

            print(f"Episode: {episode} \n Reward: {reward_for_episode} \n Average Reward: {last_reward_mean} \n Epsilon: {self.epsilon}")

    def save(self, name):
        self.model.save(name)

if __name__ == '__main__':
    env = gym.make('BipedalWalker-v3')

    lr = 0.001
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma = 0.99
    training_episodes = 2000

    '''Use this when training model'''
    model = DeepQNetwork(env, epsilon, gamma,lr)
    model.learn(training_episodes)

    '''Use this to test the model'''
    # reward_list = run_already_trained_model('LL1_model.h5', 2)
    # print(reward_list)
    env.close()
