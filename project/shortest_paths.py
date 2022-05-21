from typing import List, Tuple
import numpy as np
from pygraphblas import Matrix, Vector
from pygraphblas.types import INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64

INF = 1e10


def sssp(graph: Matrix, start_vertex: int) -> List[int]:
    """
    Single-source shortest path of a directed graph without negative cycles from a given vertex
    Parameters
    ----------
    graph: Matrix
        adjacency matrix of graph
    start_vertex: int
        number of first vertex (from 0 to n-1)
    Returns
    -------
    List[int]
        an array, where for each vertex the weight
        of the shortest path to it from the start
        vertex is indicated.
        If the vertex is not reachable, then the
        value of the corresponding cell is -1.
    """
    if not graph.square:
        raise ValueError("Adjacency matrix of the graph must be square")
    if start_vertex >= graph.nrows or start_vertex < 0:
        raise ValueError("No vertex with such number")
    if graph.type not in {INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64}:
        raise ValueError(f"Unsupported graph type: {graph.type}. Expected type: INT")
    n = graph.ncols
    d = Vector.dense(INT64, n, fill=INF)
    d[start_vertex] = 0

    for i in range(n - 1):
        d.vxm(graph, INT64.min_plus, out=d, accum=INT64.min)

    prev = np.copy(d)
    d.vxm(graph, INT64.min_plus, out=d, accum=INT64.min)
    if not np.array_equal(d, prev):
        raise ValueError("There is a negative cycle in the graph")

    d[d == INF] = -1

    return list(d.vals)


def mssp(graph: Matrix, start_vertices: List[int]) -> List[Tuple[int, List[int]]]:
    """
    Multi-source shortest path of a directed graph without negative cycles from a given vertex
    Parameters
    ----------
    graph: Matrix
        adjacency matrix of graph
    start_vertices: List[int]
        array of start vertex numbers (each from 0 to n-1)
    Returns
    -------
    List[Tuple[int, List[int]]]
        array of pairs: a vertex, and an array, where
        for each vertex the weight of the shortest path
        to it from the start vertex is indicated.
        If the vertex is not reachable, then the
        value of the corresponding cell is -1.
    """
    if not graph.square:
        raise ValueError("Adjacency matrix of the graph must be square")
    if any(
        start_vertex >= graph.nrows or start_vertex < 0
        for start_vertex in start_vertices
    ):
        raise ValueError("No vertex with such number")
    if graph.type not in {INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64}:
        raise ValueError(f"Unsupported graph type: {graph.type}. Expected type: INT")
    n = graph.ncols
    d = Matrix.dense(INT64, ncols=n, nrows=len(start_vertices), fill=INF)
    for i, start_vertex in enumerate(start_vertices):
        d.assign_scalar(0, i, start_vertex)

    for i in range(n - 1):
        d.mxm(graph, INT64.min_plus, out=d, accum=INT64.min)

    prev = np.copy(d)
    d.mxm(graph, INT64.min_plus, out=d, accum=INT64.min)
    if not np.array_equal(d, prev):
        raise ValueError("There is a negative cycle in the graph")

    d[d == INF] = -1

    return [(vertex, list(d[i].vals)) for i, vertex in enumerate(start_vertices)]
