from grid import Grid
import numpy as np

class GridWrapper(Grid):
    def __init__(self, N):
        Grid.__init__(self, N)
        self.state_size = N * N
        self.action_size = len(self.moves)

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
        assert action_num in self.get_available_moves()

        self.play(action_num)
        self.add()
        return self.curr_state(), self.curr_score(), self.is_game_over(), None
