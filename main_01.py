# -*- coding: utf-8 -*-
## todo: Code largely copied from https://github.com/keon/deep-q-learning
import os.path
import random
import numpy as np
import sys
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import logging

from grid_wrapper import GridWrapper

EPISODES = 1000
SAVE_DIR = "./save/"
DNN_FILE = SAVE_DIR + "2048-dqn.h5"

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=3000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, illegal_moves):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        for move in illegal_moves:
            act_values[0][move] = np.NINF

        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    # LOG
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger('2048-dqn')
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    hdlr = logging.FileHandler(SAVE_DIR + '2048-dqn.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    logger.info("\n\n Starting training...")

    # Initialize
    env = GridWrapper(4) #gym.make('CartPole-v1')
    state_size = env.state_size #env.observation_space.shape[0]
    action_size = env.action_size #env.action_space.n
    agent = DQNAgent(state_size, action_size)
    if os.path.isfile(DNN_FILE):
        agent.load(DNN_FILE)
    done = False
    batch_size = 200

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        max_num_moves = 10000
        for time in range(max_num_moves):
            # env.show()
            action = agent.act(state, env.get_available_moves())
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                logger.info("episode: {}/{}, score: {}, e: {:.2}, maxtile: {}"
                            .format(e, EPISODES, time, agent.epsilon, env.curr_score()))
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if e % 10 == 0:
            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)
            if not os.path.exists(DNN_FILE):
                f = open(DNN_FILE, 'x')
                f.close()
            agent.save(DNN_FILE)