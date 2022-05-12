import pytest
import pygraphblas as gb
from project.bfs import bfs, multi_source_bfs


@pytest.mark.parametrize(
    "I, J, V, size, start_vertex, expected_ans",
    [
        (
            [0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6],
            [1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4],
            [True] * 12,
            7,
            0,
            [0, 1, 2, 1, 2, 3, 2],
        ),
        (
            [0, 1, 1],
            [1, 1, 2],
            [True, True, False],
            3,
            0,
            [0, 1, -1]
        ),
        (
            [0],
            [0],
            [False],
            1,
            0,
            [0]
        ),
        (
            [1, 4, 1, 2, 2, 7, 7, 7, 4, 3, 3, 5],
            [4, 1, 2, 7, 5, 4, 3, 5, 3, 6, 3, 6],
            [True] * 12,
            8,
            0,
            [0, -1, -1, -1, -1, -1, -1, -1]
        ),
        (
            [1, 4, 1, 2, 2, 7, 7, 7, 4, 3, 3, 5],
            [4, 1, 2, 7, 5, 4, 3, 5, 3, 6, 3, 6],
            [True] * 12,
            8,
            1,
            [-1, 0, 1, 2, 1, 2, 3, 2]
        ),

    ],
)
def test_bfs(I, J, V, size, start_vertex, expected_ans):
    graph = gb.Matrix.from_lists(I, J, V, nrows=size, ncols=size)
    assert bfs(graph, start_vertex) == expected_ans


@pytest.mark.parametrize(
    "I, J, V, size, start_vertices, expected_ans",
    [
        (
            [0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6],
            [1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4],
            [True] * 12,
            7,
            [0],
            [(0, [0, 1, 2, 1, 2, 3, 2])],
        ),
        (
            [0, 1, 1],
            [1, 1, 2],
            [True, True, False],
            3,
            [0, 1, 2],
            [(0, [0, 1, -1]), (1, [-1, 0, -1]), (2, [-1, -1, 0])]
        ),
        (
            [0],
            [0],
            [False],
            1,
            [0],
            [(0, [0])]
        ),
        (
            [1, 4, 1, 2, 2, 7, 7, 7, 4, 3, 3, 5],
            [4, 1, 2, 7, 5, 4, 3, 5, 3, 6, 3, 6],
            [True] * 12,
            8,
            [0, 1, 2],
            [
                (0, [0, -1, -1, -1, -1, -1, -1, -1]),
                (1, [-1, 0, 1, 2, 1, 2, 3, 2]),
                (2, [-1, 3, 0, 2, 2, 1, 2, 1])
            ]
        ),
    ],
)
def test_multi_source_bfs(I, J, V, size, start_vertices, expected_ans):
    graph = gb.Matrix.from_lists(I, J, V, nrows=size, ncols=size)
    assert multi_source_bfs(graph, start_vertices) == expected_ans


def test_non_square_bfs():
    adj_matrix = gb.Matrix.dense(gb.BOOL, nrows=5, ncols=7)
    with pytest.raises(ValueError):
        bfs(adj_matrix, 0)


def test_non_square_multi_source_bfs():
    adj_matrix = gb.Matrix.dense(gb.BOOL, nrows=5, ncols=7)
    with pytest.raises(ValueError):
        multi_source_bfs(adj_matrix, [0])


def test_invalid_matrix_type_bfs():
    adj_matrix = gb.Matrix.dense(gb.INT64, nrows=5, ncols=5)
    with pytest.raises(ValueError):
        bfs(adj_matrix, 0)


def test_invalid_matrix_type_multi_source_bfs():
    adj_matrix = gb.Matrix.dense(gb.INT64, nrows=5, ncols=5)
    with pytest.raises(ValueError):
        multi_source_bfs(adj_matrix, [0])


def test_invalid_start_vertex_bfs():
    adj_matrix = gb.Matrix.dense(gb.BOOL, nrows=5, ncols=5)
    with pytest.raises(ValueError):
        bfs(adj_matrix, 5)


def test_invalid_start_vertex_multi_source_bfs():
    adj_matrix = gb.Matrix.dense(gb.BOOL, nrows=5, ncols=5)
    with pytest.raises(ValueError):
        multi_source_bfs(adj_matrix, [5])
