import math
from typing import List, Tuple
import numpy as np
from pygraphblas import Matrix, Vector
from pygraphblas.types import (
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    FP32,
    FP64,
)


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
    if graph.type not in {
        INT8,
        INT16,
        INT32,
        INT64,
        UINT8,
        UINT16,
        UINT32,
        UINT64,
        FP32,
        FP64,
    }:
        raise ValueError(
            f"Unsupported graph type: {graph.type}. Expected type: INT or FLOAT"
        )
    n = graph.ncols
    print(max(graph.vals))
    if graph.type in {FP32, FP64} or max(graph.vals) * (n - 1) > 2 ** 62 - 1:
        inf = math.inf
        d = Vector.dense(FP64, n, fill=inf)
    else:
        inf = 2 ** 62
        d = Vector.dense(INT64, n, fill=inf)
    d[start_vertex] = 0

    for i in range(n):
        prev = np.copy(d)
        d.vxm(graph, d.type.min_plus, out=d, accum=d.type.min)
        if np.array_equal(d, prev):
            break
        elif i == n - 1:
            raise ValueError("There is a negative cycle in the graph")

    d[d == inf] = -1

    return list(d.vals)


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
    if not graph.square:
        raise ValueError("Adjacency matrix of the graph must be square")
    if any(
        start_vertex >= graph.nrows or start_vertex < 0
        for start_vertex in start_vertices
    ):
        raise ValueError("No vertex with such number")
    if graph.type not in {
        INT8,
        INT16,
        INT32,
        INT64,
        UINT8,
        UINT16,
        UINT32,
        UINT64,
        FP32,
        FP64,
    }:
        raise ValueError(
            f"Unsupported graph type: {graph.type}. Expected type: INT or FLOAT"
        )

    return [(vertex, sssp(graph, vertex)) for vertex in start_vertices]
