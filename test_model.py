import logging
import os

import numpy as np
import sys

from DDQNAgent import DDQNAgent
from grid import Grid

def eval_agent(agent, env, NO_OF_RUNS):
    max_tiles = {}
    scores = 0
    move_counts = 0
    for _ in range(NO_OF_RUNS):
        env.reset()
        move_count = 0
        curr_score = 0
        while not env.is_game_over():
            state = np.reshape(env.get_curr_state(), [1, env.state_size])
            move = agent.act_policy(state, env.get_available_moves())
            curr_score += env.standard_score_wo_lose_penalty(move)
            move_count += 1
            env.play(move)
            env.add()
        scores += curr_score
        move_counts += move_count

        max_tile = env.get_max_tile()
        if max_tile not in max_tiles:
            max_tiles[max_tile] = 0
        max_tiles[max_tile] += 1

    print("max tiles:", sorted(max_tiles.items()))
    print("avg score:", scores/NO_OF_RUNS)
    print("avg moves:", move_counts/NO_OF_RUNS)

if __name__ == '__main__':
    WEIGHTS_FILE = 'result/2048-ddqn-sparse-32-0.95-0.0001-lp-de-True.h5'
    EXPERIMENT_NAME = WEIGHTS_FILE.split('/')[1]
    NO_OF_RUNS = 100

    env = Grid(4)
    agent = DDQNAgent(env.state_size, env.action_size)
    agent.load(WEIGHTS_FILE)

    eval_agent(agent, env, NO_OF_RUNS)