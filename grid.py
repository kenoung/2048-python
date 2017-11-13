
import numpy as np

WINNING_NUMBER = 2048

class Grid(object):
    def __init__(self, N):
        """Initialize an N*N grid"""
        self.score = 0
        self.N = N
        self.mat = np.zeros((N,N))

        self.moves = {
            'UP': self.up,
            'DOWN': self.down,
            'LEFT': self.left,
            'RIGHT': self.right
        }

    def add(self):
        empty_cells = np.argwhere(self.mat == 0)
        x, y = empty_cells[np.random.choice(len(empty_cells))]
        self.mat[x, y] = np.random.choice([2, 4], p=[0.9, 0.1])

    def show(self):
        print(self.mat)

    def get_available_moves(self):
        available_moves = []

        for move, move_fn in self.moves.items():
            if not np.array_equal(move_fn(self.mat), self.mat):
                available_moves.append(move)

        return available_moves

    def is_game_over(self):
        return len(self.get_available_moves()) == 0

    def is_win(self):
        return WINNING_NUMBER in self.mat

    def get_ele(self, i, j):
        return self.mat[i, j]


    #########
    # Moves #
    #########
    def play(self, move):
        """plays the given move
        :param string move: one of 'UP', 'DOWN', 'LEFT', 'RIGHT'
        """
        self.mat = self.moves[move](self.mat)

    def up(self, mat):
        return np.apply_along_axis(self._left, 0, mat)

    def down(self, mat):
        rev_mat = np.flipud(mat)
        shifted_mat = self.up(rev_mat)
        return np.flipud(shifted_mat)

    def left(self, mat):
        return np.apply_along_axis(self._left, 1, mat)

    def right(self, mat):
        rev_mat = np.fliplr(mat)
        shifted_mat = self.left(rev_mat)
        return np.fliplr(shifted_mat)

    def _left(self, arr):
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









