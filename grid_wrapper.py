from grid import Grid
import numpy as np

class GridWrapper(Grid):
    def __init__(self, N):
        Grid.__init__(self, N)
        self.state_size = N * N
        self.actions = [self.up, self.down, self.left, self.right]
        self.action_size = len(self.actions)

    def get_available_moves(self):
        available_moves = []

        for i in range(self.action_size):
            if not np.array_equal(self.actions[i](self.mat), self.mat):
                available_moves.append(i)
        return available_moves

    def curr_score(self):
        return self.mat.max()

    def curr_state(self):
        return self.mat.flatten()

    def reset(self):
        self.__init__(self.N)
        self.add()
        self.add()
        return self.curr_state()

    def step(self, action_num):
        # Should not happen
        if action_num not in self.get_available_moves():
            return self.curr_state(), 0, False, None

        self.mat = self.actions[action_num](self.mat)
        self.add()
        return self.curr_state(), self.curr_score(), self.is_game_over(), None
