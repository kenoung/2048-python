import numpy as np
import pytest
from grid import Grid

@pytest.mark.parametrize("N, expected_shape, expected_empty", [
    (1, (1,1), 0),
    (2, (2,2), 3),
    (4, (4,4), 15),
])
def test_grid_init(N, expected_shape, expected_empty):
    grid = Grid(N)
    assert grid.mat.shape == expected_shape
    assert len(np.argwhere(grid.mat == 0)) == expected_empty
    assert 2 in grid.mat or 4 in grid.mat



@pytest.mark.parametrize("input_arr, expected_arr", [
    ([0, 0, 0, 0], [0, 0, 0, 0]),
    ([2, 0, 0, 0], [2, 0, 0, 0]),
    ([0, 2, 0, 0], [2, 0, 0, 0]),
    ([0, 0, 0, 2], [2, 0, 0, 0]),
    ([2, 2, 0, 0], [4, 0, 0, 0]),
    ([2, 0, 0, 2], [4, 0, 0, 0]),
    ([2, 0, 2, 2], [4, 2, 0, 0]),
    ([2, 2, 2, 2], [4, 4, 0, 0]),
    ([2, 2, 0, 2], [4, 2, 0, 0]),
    ([2, 2, 4, 0], [4, 4, 0, 0]),
])
def test_left(input_arr, expected_arr):
    grid = Grid(4)
    assert grid._left(input_arr) == expected_arr


@pytest.mark.parametrize("input_arr, expected_arr", [
    ([[0, 0, 0, 0],
      [2, 0, 0, 0],
      [0, 2, 0, 0],
      [0, 0, 0, 2]],
     [[0, 0, 0, 0],
      [2, 0, 0, 0],
      [2, 0, 0, 0],
      [2, 0, 0, 0]])
])
def test_left(input_arr, expected_arr):
    grid = Grid(4)
    grid.mat = input_arr
    grid.play('LEFT')
    assert np.array_equal(grid.mat, expected_arr)


@pytest.mark.parametrize("N, no_available_moves", [
    (1, 0),
    (4, 4)
])
def test_available_moves(N, no_available_moves):
    grid = Grid(N)
    assert len(grid.get_available_moves()) == no_available_moves


@pytest.mark.parametrize("N, is_game_over", [
    (1, True),
    (4, False)
])
def test_lose(N, is_game_over):
    grid = Grid(N)
    assert grid.game_over() == is_game_over


