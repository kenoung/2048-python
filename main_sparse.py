# -*- coding: utf-8 -*-
## todo: Code largely adapted from https://github.com/keon/deep-q-learning
## todo: Analysis inspired by https://jaromiru.com/2016/10/12/lets-make-a-dqn-debugging/
import argparse
import os.path
import numpy as np
import sys
import logging

import time

from DQNAgent import DECAYING_EPSILON
from DDQNAgent import DDQNAgent
from grid import Grid, LOSE_PENALTY


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
    parser.add_argument('--reward_func', type=str)
    parser.add_argument('--epsilon_func', type=str)

    return vars(parser.parse_args())


def evaluate(agent, N, logger, reward_func_str):
    env = Grid(4)
    env.set_reward(reward_func_str)
    max_tile_distribution = {}
    overall_sum_moves = 0
    overall_sum_q = 0

    state_first = [[0,  0,  0,  0],
                   [0,  0,  0,  0],
                   [2,  0,  2,  0],
                   [0,  0,  0,  0]]
    state_1 = [[2, 0, 4,   2],
               [0, 0, 0,  16],
               [0, 0, 8,  32],
               [0, 4, 32, 16]]
    state_2 = [[2,   2,  4,  2],
               [8,  16,  0,  0],
               [64, 32,  0,  4],
               [64,  0,  0,  0]]
    state_3 = [[2, 32, 128,  2],
               [8, 128, 64, 16],
               [8, 32,  16,  8],
               [2,  8,   4,  2]]
    state_4 = [[2,   32,  128,   2],
               [8,  128,   64,  16],
               [8,   32,   16,   8],
               [2,    8,    4,   2]]
    state_5 = [[4,  2,   16, 256],
               [4,  8,   64,  32],
               [0,  0,  128,   4],
               [0,  0,  256,   8]]
    state_6 = [[4, 4,  16, 256],
               [2, 2, 512,  32],
               [0, 8, 128,   4],
               [0, 0,  16,   8]]
    state_last = [[32,       0,    0,  0],
                  [8,        2,    0,  0],
                  [64,      16,    0,  16],
                  [1024,  1024,    4,  4]]
    state_after = [[8,  32,   512,  32],
                   [8,  16,  1024,  16],
                   [64,  0,     8,  512],
                   [0,   0,     0,  2048]]
    state_right = [[0, 0,  2, 32],
                  [0, 0, 32, 16],
                  [2, 0,  8, 64],
                  [0, 0,  0, 0]]
    state_up = [[32, 16, 64, 0],
                [2, 32, 16, 0],
                [0, 0, 0, 0],
                [0, 0, 2, 0]]
    state_left = [[0,   0, 0, 0],
                  [64, 16, 0, 2],
                  [16, 32, 0, 0],
                  [32,  2, 0, 0]]
    state_down = [[0, 2, 0, 0],
                  [0, 0, 0, 0],
                  [0, 16, 32, 2],
                  [0, 64, 16, 32]]
    states = [state_first, state_1, state_2, state_3, state_4, state_5, state_6, state_last, state_after, state_right, state_up, state_left, state_down]
    states_names = ["state_first", "state_1", "state_2", "state_3", "state_4", "state_5", "state_6", "state_last", "state_after", "state_right", "state_up", "state_left", "state_down"]
    states = list(map(lambda x: np.ma.log2(np.array(x).flatten()).filled(0) / 10, states))

    results = {}
    for i in range(len(states)):
        state = states[i]
        state_name = states_names[i]
        q = agent.gamma * np.amax(agent.model.predict(np.reshape(state, [1, env.state_size]))[0])
        action = env.moves[np.argmax(agent.model.predict(np.reshape(state, [1, env.state_size]))[0])].__name__
        results[state_name] = {"q":q,"action":action}
    logger.info(str(results))

    for i in range(N):
        env.reset()
        moves = 0
        sum_q = 0

        while not env.is_game_over():
            state = np.reshape(env.get_curr_state(), [1, env.state_size])
            action = agent.act_policy(state, env.get_available_moves())
            reward = env.reward_func(action)
            env.play(action)
            env.add()
            moves += 1

            if env.is_game_over():
                reward = env.reward_func(action)

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
    logger.info("Performance at episode: {}, max tile distribution: {}, avg no of moves: {}, mean q value: {}"
                .format(e, sorted(max_tile_distribution.items(), key=lambda x: x[0]), overall_sum_moves/N, overall_sum_q/overall_sum_moves))


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
    REWARD_FUNC = PARAMS.get('reward_func') or LOSE_PENALTY
    EPSILON_FUNC = PARAMS.get('epsilon_func') or DECAYING_EPSILON

    EPISODES = 1000000
    SAVE_DIR = "./save/"
    EXPERIMENT_NAME = "2048-ddqn-sparse-{}-{}-{}-{}-{}".format(BATCH_SIZE, GAMMA, LR, REWARD_FUNC, EPSILON_FUNC)
    DNN_FILE = SAVE_DIR + EXPERIMENT_NAME + ".h5"
    LOG_FILE = SAVE_DIR + EXPERIMENT_NAME + ".log"

    logger = init_logger()

    # Initialize
    env = Grid(4)
    env.set_reward(REWARD_FUNC)
    agent = DDQNAgent(env.state_size, env.action_size, GAMMA, LR)
    agent.set_epsilon(EPSILON_FUNC)
    if os.path.isfile(DNN_FILE):
        logger.info('loading file from {}'.format(DNN_FILE))
        agent.load(DNN_FILE)
    max_num_moves = 10000
    logger.info("gamma = {}, epsilon = {}, epsilon_min = {}, epsilon_decay = {}, learning_rate = {}"
                .format(agent.gamma, agent.epsilon, agent.epsilon_min, agent.epsilon_decay, agent.learning_rate))
    logger.info("reward_func = {}, epsilon_func = {}".format(REWARD_FUNC, EPSILON_FUNC))
    logger.info("batch_size = {}, memory_size = {}, max_num_moves = {}"
                .format(BATCH_SIZE, agent.memory.maxlen, max_num_moves))

    # Train
    overall_start_time = time.time()
    for e in range(EPISODES):
        env.reset()
        state = np.reshape(env.get_curr_state(), [1, env.state_size])

        if e % 1000 == 0:
            agent.update_target_model()
            evaluate(agent, 10, logger, REWARD_FUNC)
            save_weights(agent, SAVE_DIR, DNN_FILE)

        t0 = time.time()
        for t in range(max_num_moves):
            action = agent.act(state, env.get_available_moves())
            reward = env.reward_func(action)
            env.play(action)
            env.add()
            next_state = np.reshape(env.get_curr_state(), [1, env.state_size])

            if not env.is_game_over():
                agent.remember(state, action, reward, next_state, env.is_game_over())
                state = next_state
            else:
                reward = env.reward_func(action) # Get gameover reward
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
