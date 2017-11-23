# -*- coding: utf-8 -*-
## todo: Code largely adapted from https://github.com/keon/deep-q-learning
## todo: Parameters largely adapted from https://github.com/georgwiese/2048-rl
import uuid

import os.path
import random
import numpy as np
import sys
import time as tm
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.optimizers import Adam
import logging

from grid_wrapper_replicate import GridWrapper

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999995
        self.learning_rate = 0.01
        self.model = self._build_model()

    def deduplicate(self, experiences):
        """Returns a new experience array that contains contains no duplicates."""

        state_set = set()
        filterted_experiences = []
        for experience in experiences:
            state_tuple = tuple(experience[0].flatten())
            if not state_tuple in state_set:
                state_set.add(state_tuple)
                filterted_experiences.append(experience)
        return filterted_experiences

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='relu'))
        model.add(SimpleRNN(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, legal_moves):
        if np.random.rand() <= self.epsilon:
            return random.choice(legal_moves)

        act_values = self.model.predict(state)
        for move in range(action_size):
            if move not in legal_moves:
                act_values[0][move] = np.NINF

        return np.argmax(act_values[0])  # returns action

    def act_policy(self, state, legal_moves):
        act_values = self.model.predict(state)
        for move in range(action_size):
            if move not in legal_moves:
                act_values[0][move] = np.NINF

        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        time_diff = 0
        minibatch = self.deduplicate(self.memory)
        minibatch = random.sample(minibatch, batch_size)
        for state, action, reward, next_state, done in minibatch:
            start = tm.time()
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            end = tm.time()
            time_diff += end - start
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return time_diff

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":

    EPISODES = 1000000
    SAVE_DIR = "./save/"
    EXPERIMENT_NAME = "2048-dqn-replicate-" + str(uuid.uuid4())
    DNN_FILE = SAVE_DIR + EXPERIMENT_NAME + ".h5"
    LOG_FILE = SAVE_DIR + EXPERIMENT_NAME + ".log"

    # LOG
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger(EXPERIMENT_NAME)
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    hdlr = logging.FileHandler(LOG_FILE)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    logger.info("\n\n Starting training...")

    # Initialize
    env = GridWrapper(4)
    state_size = env.state_size
    action_size = env.action_size
    agent = DQNAgent(state_size, action_size)
    if os.path.isfile(DNN_FILE):
        agent.load(DNN_FILE)
    done = False
    max_num_moves = 10000
    batch_size = 32
    logger.info("gamma = {}, epsilon = {}, epsilon_min = {}, epsilon_decay = {}, learning_rate = {}"
                .format(agent.gamma, agent.epsilon, agent.epsilon_min, agent.epsilon_decay, agent.learning_rate))
    logger.info("batch_size = {}, memory_size = {}, max_num_moves = {}"
                .format(batch_size, agent.memory.maxlen, max_num_moves))

    overall_start_time = tm.time()
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        moves = 0
        episode_time_start = tm.time()
        for time in range(max_num_moves):
            # env.show()
            action = agent.act(state, env.get_available_moves())
            next_state, reward, done, _ = env.step(action)

            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                moves = time
                break

        simulation_time = tm.time() - episode_time_start

        training_time = 0.0
        if len(agent.memory) > batch_size:
            training_time = agent.replay(batch_size)

        if e % 10 == 0:
            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)
            if not os.path.exists(DNN_FILE):
                f = open(DNN_FILE, 'x')
                f.close()
            agent.save(DNN_FILE)

        episode_time = tm.time() - episode_time_start
        logger.info("episode: {}/{}, moves: {}, e: {:.2}, maxtile: {}, sim_time: {:.3}, train_time: {:.3}, episode_time: {:.3}, score: {}"
                    .format(e, EPISODES, moves, agent.epsilon, env.get_max_tile(), simulation_time, training_time, episode_time, env.score))

        # Play 1000 games
        if e % 10000 == 0:
            game = GridWrapper(4)
            for i in range(1, 101):
                state = game.reset()
                state = np.reshape(state, [1, state_size])
                done = False
                moves = 0
                mean_q = 0

                while not done:
                    # env.show()
                    action = agent.act_policy(state, game.get_available_moves())
                    next_state, _, done, _ = game.step(action)
                    next_state = np.reshape(next_state, [1, state_size])
                    state = next_state
                    moves += 1

                    q = reward
                    if not done:
                        q = (reward + agent.gamma * np.amax(agent.model.predict(next_state)[0]))
                    mean_q += q

                mean_q /= moves

                logger.info(
                    "Playing at episode: {}, game num: {}, moves: {}, maxtile: {}, score: {}, mean_q: {}"
                    .format(e, i, moves, game.get_max_tile(), game.score, mean_q))

    overall_time = tm.time() - overall_start_time
    logger.info("total time taken: {:.3}".format(overall_time))


# Changes made:
# Implement deduplicate
# Implement mean Q
# Implement RNN