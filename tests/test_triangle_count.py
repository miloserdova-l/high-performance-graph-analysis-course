import pytest
from pygraphblas import Matrix
from pygraphblas.types import BOOL, INT64
from project.triangle_count import triangle_count


@pytest.mark.parametrize(
    "I, J, V, size, expected_ans",
    [
        (
            [0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6],
            [1, 3, 0, 3, 4, 6, 3, 5, 6, 0, 1, 2, 5, 6, 1, 5, 6, 2, 3, 4, 1, 2, 3, 4],
            [True] * 24,
            7,
            [1, 3, 2, 4, 1, 1, 3],
        ),
        (
            [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
            [1, 3, 3, 4, 6, 3, 5, 6, 5, 6, 5, 6],
            [True] * 12,
            7,
            [1, 3, 2, 4, 1, 1, 3],
        ),
        (
            [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
            [1, 3, 3, 4, 6, 3, 5, 6, 5, 6, 5, 6],
            [True] * 12,
            8,
            [1, 3, 2, 4, 1, 1, 3, 0],
        ),
    ],
)
def test_triangle_count(I, J, V, size, expected_ans):
    graph = Matrix.from_lists(I, J, V, nrows=size, ncols=size)
    assert triangle_count(graph) == expected_ans


def test_non_square_triangle_count():
    graph = Matrix.dense(BOOL, nrows=5, ncols=7)
    with pytest.raises(ValueError):
        triangle_count(graph)


def test_invalid_matrix_type_triangle_count():
    graph = Matrix.dense(INT64, nrows=5, ncols=5)
    with pytest.raises(ValueError):
        triangle_count(graph)
