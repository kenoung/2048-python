import random

import time
from copy import deepcopy

from grid import Grid


def get_result_from_random_rollout(tempenv, depth):
    steps = 0
    while not tempenv.is_game_over():
        steps += 1
        move = random.choice(tempenv.get_available_moves())
        tempenv.play(move)
        tempenv.add()

        if steps > depth:
            return steps, tempenv.get_max_tile()

    return steps, tempenv.get_max_tile()


def get_best_move(tempenv, mat, no_of_rollout, depth):
    best_action, best_steps, best_maxtile = None, -1, -1

    tempenv.reset()
    tempenv.mat = deepcopy(mat)
    for action in tempenv.get_available_moves():
        tempenv.reset()
        tempenv.mat = deepcopy(mat)
        tempenv.play(action)
        tempenv.add()
        tempmat = tempenv.mat

        steps = 0
        maxtiles = 0

        if not tempenv.is_game_over():
            for _ in range(no_of_rollout):
                tempenv.reset()
                tempenv.mat = deepcopy(tempmat)
                step, maxtile = get_result_from_random_rollout(tempenv, depth)
                steps += step
                maxtiles += maxtile

        if steps > best_steps or maxtiles > best_maxtile:
            best_action, best_steps, best_maxtile = action, steps, maxtiles

    print('choose {}'.format(best_action))
    return best_action

if __name__ == '__main__':
    env = Grid(4)
    env.winning_number = 4
    tempenv = Grid(4)
    tempenv.winning_number = 4

    NO_OF_TRIALS = 1000
    NO_OF_ROLLOUT = 20
    DEPTH = 100
    best_move_dict = {}

    for trial in range(NO_OF_TRIALS):
        env.reset()

        while not env.is_game_over():
            t1 = time.time()
            move = get_best_move(tempenv, env.mat, NO_OF_ROLLOUT, DEPTH)
            print(time.time() - t1)
            env.show()
            env.play(move)
            env.add()


        print(env.get_max_tile())

