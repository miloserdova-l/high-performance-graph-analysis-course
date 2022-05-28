from typing import List, Tuple
from pygraphblas import Matrix, Vector
from pygraphblas.types import BOOL, INT64
from pygraphblas.descriptor import R, RC


def bfs(graph: Matrix, start_vertex: int) -> List[int]:
    """
    Breadth-first search of a directed graph from a given vertex
    Parameters
    ----------
    graph: Matrix
        adjacency matrix of graph
    start_vertex: int
        number of first vertex (from 0 to n-1)
    Returns
    -------
    List[int]
        an array, where for each vertex it is indicated
        at what step it is reachable. The start vertex is reachable
        at the zero step, if the vertex is not reachable, then the
        value of the corresponding cell is -1.
    """
    if not graph.square:
        raise ValueError("Adjacency matrix of the graph must be square")
    if start_vertex >= graph.nrows or start_vertex < 0:
        raise ValueError("No vertex with such number")
    if graph.type != BOOL:
        raise ValueError(f"Unsupported graph type: {graph.type}. Expected type: BOOL")
    n = graph.ncols
    q = Vector.sparse(BOOL, n)
    used = Vector.sparse(BOOL, n)
    answer = Vector.dense(INT64, n, fill=-1)
    answer[start_vertex] = 0

    used[start_vertex] = True
    q[start_vertex] = True
    step = 1
    prev_nnz = -1
    while used.nvals != prev_nnz:
        prev_nnz = used.nvals
        q.vxm(graph, mask=used, desc=RC, out=q)
        used.eadd(q, BOOL.lor_land, desc=R, out=used)
        answer.assign_scalar(step, mask=q)
        step += 1

    return list(answer.vals)


def multi_source_bfs(
    graph: Matrix, start_vertices: List[int]
) -> List[Tuple[int, List[int]]]:
    """
    Breadth-first search of a directed graph from a given vertex
    Parameters
    ----------
    graph: Matrix
        adjacency matrix of graph
    start_vertices: List[int]
        array of start vertex numbers (each from 0 to n-1)
    Returns
    -------
    List[Tuple[int, List[int]]]
        array of pairs: a vertex, and an array, where for each vertex
        it is indicated at which step it is reachable from the specified one.
        The start vertex is reachable at the zero step, if the vertex is not
        reachable, then the value of the corresponding cell is -1.
    """
    if not graph.square:
        raise ValueError("Adjacency matrix of the graph must be square")
    if any(
        start_vertex >= graph.nrows or start_vertex < 0
        for start_vertex in start_vertices
    ):
        raise ValueError("No vertex with such number")
    if graph.type != BOOL:
        raise ValueError(f"Unsupported graph type: {graph.type}. Expected type: BOOL")
    n = graph.ncols
    m = len(start_vertices)
    q = Matrix.sparse(BOOL, nrows=m, ncols=n)
    used = Matrix.sparse(BOOL, nrows=m, ncols=n)
    answer = Matrix.dense(INT64, nrows=m, ncols=n, fill=-1)

    for i, j in enumerate(start_vertices):
        q.assign_scalar(True, i, j)
        used.assign_scalar(True, i, j)
        answer.assign_scalar(0, i, j)

    step = 1
    prev_nnz = -1
    while used.nvals != prev_nnz:
        prev_nnz = used.nvals
        q.mxm(graph, mask=used, out=q, desc=RC)
        used.eadd(q, BOOL.lor_land, out=used, desc=R)
        answer.assign_scalar(step, mask=q)
        step += 1

    return [(vertex, list(answer[i].vals)) for i, vertex in enumerate(start_vertices)]
