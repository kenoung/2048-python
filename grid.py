import logging

import numpy as np

import pickle

WINNING_NUMBER = 2048

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

LOSE_PENALTY = "lp"
MAX_MERGEABLE_TILES = "mmt"
STANDARD_SCORE_WO_LOSE_PENALTY = "sswolp"
STANDARD_SCORE_W_LOSE_PENALTY = "sswlp"

class Grid(object):
    def __init__(self, N):
        """Initialize an N*N grid"""
        self.N = N
        self.mat = np.zeros((N,N))
        self.state_size = N*N
        self.action_size = 4
        self.available_moves = None

        self.moves_str = {
            'UP': self.up,
            'DOWN': self.down,
            'LEFT': self.left,
            'RIGHT': self.right
        }

        self.moves = {
            UP: self.up,
            DOWN: self.down,
            LEFT: self.left,
            RIGHT: self.right
        }

        self.reward_funcs = {
            LOSE_PENALTY: self.lose_penalty,
            MAX_MERGEABLE_TILES: self.max_mergeable_tiles,
            STANDARD_SCORE_WO_LOSE_PENALTY: self.standard_score_wo_lose_penalty,
            STANDARD_SCORE_W_LOSE_PENALTY: self.standard_score_w_lose_penalty
        }
        self.reward_func = None

        self.next_arr = [None] * 4

        with open('tmap.pickle', 'rb') as f:
            self.tmap = pickle.load(f)

    def add(self):
        empty_cells = np.argwhere(self.mat == 0)
        x, y = empty_cells[np.random.choice(len(empty_cells))]
        self.mat[x, y] = np.random.choice([2, 4], p=[0.9, 0.1])

    def show(self):
        print(self.mat)

    def get_available_moves(self):
        if not self.available_moves:
            available_moves = []

            for move, move_fn in self.moves.items():
                if not np.array_equal(move_fn(self.mat), self.mat):
                    available_moves.append(move)

            self.available_moves = available_moves

        return self.available_moves

    def get_available_moves_str(self):
        available_moves = []

        for move, move_fn in self.moves_str.items():
            if not np.array_equal(move_fn(self.mat), self.mat):
                available_moves.append(move)

        return available_moves

    def is_game_over(self):
        return len(self.get_available_moves()) == 0

    def is_win(self):
        return WINNING_NUMBER in self.mat

    def get_ele(self, i, j):
        return self.mat[i, j]

    def reset(self):
        self.mat = np.zeros((self.N,self.N))
        self.available_moves = None
        self.next_arr = [None] * 4
        self.add()
        self.add()

    def get_curr_state(self):
        return np.ma.log2(self.mat.flatten()).filled(0) / 10

    def get_max_tile(self):
        return self.mat.max()

    def get_num_tiles(self, mat):
        num_tiles = 0
        for tile in mat.flatten():
            if tile != 0:
                num_tiles += 1
        return num_tiles

    #########
    # Moves #
    #########
    def play_str(self, move_str):
        """
        Plays the given move
        :param string move: one of 'UP', 'DOWN', 'LEFT', 'RIGHT'
        """
        self.mat = self.moves_str[move_str](self.mat)

    def play(self, move):
        """
        Plays the given move
        :param move_int: 0 - 3 (corresponds to 'UP', 'DOWN', 'LEFT', 'RIGHT')
        """
        self.next_arr = [None] * 4
        self.available_moves = None
        self.mat = self.moves[move](self.mat)

    def up(self, mat):
        if self.next_arr[UP] is None:
            self.next_arr[UP] = self._up(mat)
        return self.next_arr[UP]

    def down(self, mat):
        if self.next_arr[DOWN] is None:
            self.next_arr[DOWN] = self._down(mat)

        return self.next_arr[DOWN]

    def left(self, mat):
        if self.next_arr[LEFT] is None:
            self.next_arr[LEFT] = self._left(mat)

        return self.next_arr[LEFT]

    def right(self, mat):
        if self.next_arr[RIGHT] is None:
            self.next_arr[RIGHT] = self._right(mat)

        return self.next_arr[RIGHT]

    def _up(self, mat):
        return np.apply_along_axis(self._left_single_row, 0, mat)

    def _down(self, mat):
        rev_mat = np.flipud(mat)
        shifted_mat = self._up(rev_mat)
        return np.flipud(shifted_mat)

    def _left(self, mat):
        return np.apply_along_axis(self._left_single_row, 1, mat)

    def _right(self, mat):
        rev_mat = np.fliplr(mat)
        shifted_mat = self._left(rev_mat)
        return np.fliplr(shifted_mat)

    def _left_single_row(self, arr):
        try:
            return self.tmap[tuple(arr)]
        except KeyError:
            logging.error('arr not found: {}'.format(arr))
            return self._gen_left_single_row(arr)

    def _gen_left_single_row(self, arr):
        arr = [i for i in arr if i != 0]
        idx = 1
        new_arr = []
        while idx <= len(arr):
            if idx == len(arr):
                new_arr.append(arr[idx - 1])
                idx += 1
            elif arr[idx] == arr[idx-1]:
                new_arr.append(arr[idx-1]*2)
                idx += 2
            else:
                new_arr.append(arr[idx - 1])
                idx += 1

        new_arr.extend([0] * (self.N - len(new_arr)))

        return new_arr

    def _rotate_left(self, mat):
        return mat.T[::-1]

    def _rotate_right(self, mat):
        return mat[::-1].T

    ############
    ## REWARD ##
    ############
    def set_reward(self, reward_func_str):
        self.reward_func =  self.reward_funcs[reward_func_str]

    def lose_penalty(self, action):
        if self.is_game_over():
            return -1
        elif self.is_win():
            return 1
        else:
            return 0

    def max_mergeable_tiles(self, action):
        if self.is_game_over():
            return 0
        min_tiles = 16
        for _, act in self.moves.items():
            min_tiles = min(min_tiles, self.get_num_tiles(act(self.mat)))
        return self.get_num_tiles(self.mat) - min_tiles


    def standard_score_wo_lose_penalty(self, action):
        return self.standard_score(action, 0)

    def standard_score_w_lose_penalty(self, action):
        return self.standard_score(action, -1000)

    # Helper for standard score
    def standard_score(self, action, penalty):
        if self.is_game_over():
            return penalty
        score = 0

        # todo IMPLEMENT SCORE in actions, and return it here
        old_mat = self.mat.flatten()
        new_mat = self.moves[action](self.mat).flatten()

        old_mat_count = dict()
        for tile in old_mat:
            if tile not in old_mat_count:
                old_mat_count[tile] = 0
            old_mat_count[tile] += 1

        for tile in new_mat:
            if tile not in old_mat_count or old_mat_count[tile] == 0:
                score += tile
            else:
                old_mat_count[tile] -= 1

        return score
