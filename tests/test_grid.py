import numpy as np
import pytest
from grid import Grid

@pytest.mark.parametrize("N, expected_shape, expected_empty", [
    (1, (1,1), 1),
    (2, (2,2), 4),
    (4, (4,4), 16),
])
def test_grid_init(N, expected_shape, expected_empty):
    grid = Grid(N)
    assert grid.mat.shape == expected_shape
    assert len(np.argwhere(grid.mat == 0)) == expected_empty
    grid.add()
    assert len(np.argwhere(grid.mat == 0)) == expected_empty - 1
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
    grid.play(2)
    assert np.array_equal(grid.mat, expected_arr)


@pytest.mark.parametrize("input_mat, available_moves", [
    (
            [[2,16,2,4],
             [2,8,4,2],
             [2,8,4,2],
             [2,8,16,2]],
            [0, 1]),
])
def test_available_moves(input_mat, available_moves):
    grid = Grid(4)
    grid.mat = input_mat
    assert grid.get_available_moves() == available_moves


@pytest.mark.parametrize("input_mat, expected_mat", [
    (
            [[ 8,  4,  0,  0],
             [ 2,  0,  4,  0],
             [ 2,  0,  0,  0],
             [ 0,  0,  0,  0]],
            [[ 8,  4,  4,  0],
             [ 4,  0,  0,  0],
             [ 0,  0,  0,  0],
             [ 0,  0,  0,  0]],
    )
])
def test_up(input_mat, expected_mat):
    grid = Grid(4)
    assert np.array_equal(grid.up(input_mat), expected_mat)


@pytest.mark.parametrize("N, is_game_over", [
    (1, True),
    (4, False)
])
def test_lose(N, is_game_over):
    grid = Grid(N)
    grid.add()
    assert grid.is_game_over() == is_game_over


