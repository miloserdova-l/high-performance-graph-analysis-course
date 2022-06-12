from typing import List
import numpy as np
from pygraphblas import Matrix, Vector


def sssp(graph: Matrix, start_vertex: int) -> list:
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
    list
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
    if not hasattr(graph.type, "min_plus"):
        raise ValueError(
            f"Unsupported graph type: {graph.type}. Graph must have min_plus operation defined."
        )
    n = graph.ncols
    d = Vector.sparse(graph.type, size=n)
    d[start_vertex] = graph.type.default_zero

    for i in range(n):
        prev = np.copy(d)
        d.vxm(graph, d.type.min_plus, out=d, accum=d.type.min)
        if np.array_equal(d, prev):
            break
        elif i == n - 1:
            raise ValueError("There is a negative cycle in the graph")

    return [d.get(i, -1) for i in range(n)]


def mssp(graph: Matrix, start_vertices: List[int]) -> list:
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
    list
        array of pairs: a vertex, and an array, where
        for each vertex the weight of the shortest path
        to it from the start vertex is indicated.
        If the vertex is not reachable, then the
        value of the corresponding cell is -1.
    """
    return [(vertex, sssp(graph, vertex)) for vertex in start_vertices]
