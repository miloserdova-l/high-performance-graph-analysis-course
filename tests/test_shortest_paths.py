import pytest
from pygraphblas import Matrix
from pygraphblas.types import INT64, BOOL

from project.shortest_paths import sssp, mssp


@pytest.mark.parametrize(
    "I, J, V, size, start_vertex, expected_ans",
    [
        (
            [0, 3, 0, 1, 1, 3, 4, 5, 2, 6, 6, 6],
            [3, 0, 1, 6, 4, 2, 5, 2, 5, 2, 3, 4],
            [8, 2, 3, 7, 1, 4, 1, 5, 5, 1, 5, 8],
            7,
            0,
            [0, 3, 10, 8, 4, 5, 10],
        ),
        (
            [0, 3, 0, 1, 1, 3, 4, 5, 2, 6, 6, 6],
            [3, 0, 1, 6, 4, 2, 5, 2, 5, 2, 3, 4],
            [8.0, 2.0, 3.0, 7.0, 1.0, 4.0, 1.0, 5.0, 5.0, 1.0, 5.0, 8.0],
            7,
            1,
            [14.0, 0.0, 7.0, 12.0, 1.0, 2.0, 7.0],
        ),
        (
            [0, 3, 0, 1, 1, 3, 4, 5, 2, 6, 6, 6],
            [3, 0, 1, 6, 4, 2, 5, 2, 5, 2, 3, 4],
            [8, 2, 3, 7, 1, 4, 1, 5, 5, 1, 5, 8],
            8,
            0,
            [0, 3, 10, 8, 4, 5, 10, -1],
        ),
        (
            [0, 3, 0, 1, 1, 3, 4, 5, 2, 6, 6, 6],
            [3, 0, 1, 6, 4, 2, 5, 2, 5, 2, 3, 4],
            [
                8 * 2 ** 59,
                2 * 2 ** 59,
                3 * 2 ** 59,
                7 * 2 ** 59,
                1 * 2 ** 59,
                4 * 2 ** 59,
                1 * 2 ** 59,
                5 * 2 ** 59,
                5 * 2 ** 59,
                1 * 2 ** 59,
                5 * 2 ** 59,
                8 * 2 ** 59,
            ],
            7,
            0,
            [
                0,
                3 * 2 ** 59,
                10 * 2 ** 59,
                8 * 2 ** 59,
                4 * 2 ** 59,
                5 * 2 ** 59,
                10 * 2 ** 59,
            ],
        ),
    ],
)
def test_sssp(I, J, V, size, start_vertex, expected_ans):
    graph = Matrix.from_lists(I, J, V, nrows=size, ncols=size)
    assert sssp(graph, start_vertex) == expected_ans


@pytest.mark.parametrize(
    "I, J, V, size, start_vertices, expected_ans",
    [
        (
            [0, 3, 0, 1, 1, 3, 4, 5, 2, 6, 6, 6],
            [3, 0, 1, 6, 4, 2, 5, 2, 5, 2, 3, 4],
            [8, 2, 3, 7, 1, 4, 1, 5, 5, 1, 5, 8],
            7,
            [0],
            [(0, [0, 3, 10, 8, 4, 5, 10])],
        ),
        (
            [0, 3, 0, 1, 1, 3, 4, 5, 2, 6, 6, 6],
            [3, 0, 1, 6, 4, 2, 5, 2, 5, 2, 3, 4],
            [8, 2, 3, 7, 1, 4, 1, 5, 5, 1, 5, 8],
            8,
            [0, 1],
            [(0, [0, 3, 10, 8, 4, 5, 10, -1]), (1, [14, 0, 7, 12, 1, 2, 7, -1])],
        ),
    ],
)
def test_mssp(I, J, V, size, start_vertices, expected_ans):
    graph = Matrix.from_lists(I, J, V, nrows=size, ncols=size)
    assert mssp(graph, start_vertices) == expected_ans


def test_non_square_sssp():
    graph = Matrix.dense(INT64, nrows=5, ncols=7)
    with pytest.raises(ValueError):
        sssp(graph, 0)


def test_non_square_mssp():
    graph = Matrix.dense(INT64, nrows=5, ncols=7)
    with pytest.raises(ValueError):
        mssp(graph, [0])


def test_invalid_matrix_type_sssp():
    graph = Matrix.dense(BOOL, nrows=5, ncols=5)
    with pytest.raises(ValueError):
        sssp(graph, 0)


def test_invalid_matrix_type_mssp():
    graph = Matrix.dense(BOOL, nrows=5, ncols=5)
    with pytest.raises(ValueError):
        mssp(graph, [0])


def test_invalid_start_vertex_sssp():
    graph = Matrix.dense(INT64, nrows=5, ncols=5)
    with pytest.raises(ValueError):
        sssp(graph, 5)


def test_invalid_start_vertex_mssp():
    graph = Matrix.dense(INT64, nrows=5, ncols=5)
    with pytest.raises(ValueError):
        mssp(graph, [5])


def test_negative_cycle_sssp():
    graph = Matrix.from_lists(
        [0, 1, 2, 3, 1], [1, 2, 3, 1, 4], [5, -1, -1, 1, 1], nrows=5, ncols=5
    )
    with pytest.raises(ValueError):
        sssp(graph, 0)


def test_negative_cycle_mssp():
    graph = Matrix.from_lists(
        [0, 1, 2, 3, 1], [1, 2, 3, 1, 4], [5, -1, -1, 1, 1], nrows=5, ncols=5
    )
    with pytest.raises(ValueError):
        mssp(graph, [0])
