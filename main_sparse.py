# -*- coding: utf-8 -*-
## todo: Code largely adapted from https://github.com/keon/deep-q-learning
## todo: Analysis inspired by https://jaromiru.com/2016/10/12/lets-make-a-dqn-debugging/
import argparse
import os.path
import numpy as np
import sys
import logging

import time

from DQNAgent import DQNAgent
from DDQNAgent import DDQNAgent
from grid import Grid


def init_logger():
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
    return logger


def get_parameters():
    parser = argparse.ArgumentParser(description='Get agent parameters')
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--batch_size', type=int)

    return vars(parser.parse_args())


def evaluate(agent, N, logger):
    env = Grid(4)
    max_tile_distribution = {}
    overall_sum_moves = 0
    overall_sum_q = 0

    state_first = [[ 0,  0,  0,  0],
                 [ 0,  0,  0,  0],
                 [ 2,  0,  2,  0],
                 [ 0,  0,  0,  0]]
    state_last = [[2,  32, 128,  2],
                  [8, 128,  64, 16],
                  [8,  32,  16,  8],
                  [2,   8,   4,  2]]
    state_first = np.ma.log2(np.array(state_first).flatten()).filled(0) / 10
    state_last = np.ma.log2(np.array(state_last).flatten()).filled(0) / 10

    q_first = agent.gamma * np.amax(agent.model.predict(np.reshape(state_first, [1, env.state_size]))[0])
    q_last = agent.gamma * np.amax(agent.model.predict(np.reshape(state_last, [1, env.state_size]))[0])

    for i in range(N):
        env.reset()
        moves = 0
        sum_q = 0

        while not env.is_game_over():
            state = np.reshape(env.get_curr_state(), [1, env.state_size])
            action = agent.act_policy(state, env.get_available_moves())
            env.play(action)
            env.add()
            moves += 1

            if env.is_game_over():
                reward = 1 if env.is_win() else -1
            else:
                reward = 0

            q = reward + agent.gamma * np.amax(agent.model.predict(np.reshape(env.get_curr_state(), [1, env.state_size]))[0])
            sum_q += q

        if env.get_max_tile() not in max_tile_distribution:
            max_tile_distribution[env.get_max_tile()] = 1
        else:
            max_tile_distribution[env.get_max_tile()] += 1

        overall_sum_moves += moves
        overall_sum_q += sum_q

        logger.info(
            "Playing at episode: {}, game num: {}, moves: {}, maxtile: {}, mean q value: {}"
                .format(e, i + 1, moves, env.get_max_tile(), sum_q/moves))
    logger.info("Performance at episode: {}, max tile distribution: {}, avg no of moves: {}, mean q value: {}, first q: {}, last q: {}"
                .format(e, sorted(max_tile_distribution.items(), key=lambda x: x[0]), overall_sum_moves/N, overall_sum_q/overall_sum_moves, q_first, q_last))


def save_weights(agent, save_dir, file_path):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(file_path):
        f = open(file_path, 'x')
        f.close()
    agent.save(file_path)


if __name__ == "__main__":
    PARAMS = get_parameters()
    BATCH_SIZE = PARAMS['batch_size'] or 32
    GAMMA = PARAMS.get('gamma') or 0.95
    LR = PARAMS.get('learning_rate') or 0.001

    EPISODES = 1000000
    SAVE_DIR = "./save/"
    EXPERIMENT_NAME = "2048-ddqn-sparse-{}-{}-{}".format(BATCH_SIZE, GAMMA, LR)
    DNN_FILE = SAVE_DIR + EXPERIMENT_NAME + ".h5"
    LOG_FILE = SAVE_DIR + EXPERIMENT_NAME + ".log"

    logger = init_logger()

    # Initialize
    env = Grid(4)
    agent = DDQNAgent(env.state_size, env.action_size, GAMMA, LR)
    if os.path.isfile(DNN_FILE):
        logger.info('loading file from {}'.format(DNN_FILE))
        agent.load(DNN_FILE)
    max_num_moves = 10000
    logger.info("gamma = {}, epsilon = {}, epsilon_min = {}, epsilon_decay = {}, learning_rate = {}"
                .format(agent.gamma, agent.epsilon, agent.epsilon_min, agent.epsilon_decay, agent.learning_rate))
    logger.info("batch_size = {}, memory_size = {}, max_num_moves = {}"
                .format(BATCH_SIZE, agent.memory.maxlen, max_num_moves))

    overall_start_time = time.time()
    for e in range(EPISODES):
        env.reset()
        state = np.reshape(env.get_curr_state(), [1, env.state_size])

        if e % 1000 == 0:
            agent.update_target_model()
            evaluate(agent, 10, logger)
            save_weights(agent, SAVE_DIR, DNN_FILE)

        t0 = time.time()
        for t in range(max_num_moves):
            action = agent.act(state, env.get_available_moves())
            env.play(action)
            env.add()
            next_state = np.reshape(env.get_curr_state(), [1, env.state_size])

            if not env.is_game_over():
                agent.remember(state, action, 0, next_state, env.is_game_over())
                state = next_state
            else:
                reward = 1 if env.is_win() else -1
                agent.remember(state, action, reward, next_state, env.is_game_over())
                break

        t1 = time.time()

        loss = None
        if len(agent.memory) > BATCH_SIZE:
            loss = agent.replay(BATCH_SIZE)

        t2 = time.time()

        simulation_time = t1-t0
        training_time = t2-t1
        episode_time = t2-t0

        logger.info("episode: {}/{}, e: {:.2}, maxtile: {}, sim_time: {:.3}, train_time: {:.3}, episode_time: {:.3}, loss: {:.3}"
                    .format(e, EPISODES, agent.epsilon, env.get_max_tile(), simulation_time, training_time, episode_time, loss))


    overall_time = time.time() - overall_start_time
    logger.info("total time taken: {:.3}".format(overall_time))
