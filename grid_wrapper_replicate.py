from grid import Grid
import numpy as np

class GridWrapper(Grid):
    def __init__(self, N):
        Grid.__init__(self, N)
        self.state_size = N * N
        self.action_size = len(self.moves)
        self.score = 0

    def get_num_tiles(self):
        num_tiles = 0
        for tile in self.curr_state():
            if tile != 0:
                num_tiles += 1
        return num_tiles

    def get_max_tile(self):
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
        diff = self.get_num_tiles()
        self.play(action_num)
        self.add()
        diff -= self.get_num_tiles()
        self.score += diff
        return self.curr_state(), diff, self.is_game_over(), None
