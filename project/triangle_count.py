from typing import List
from pygraphblas import Matrix
from pygraphblas.types import BOOL, INT64


def triangle_count(graph: Matrix) -> List[int]:
    """
    Triangle count
    Parameters
    ----------
    graph: Matrix
        adjacency matrix of graph
    Returns
    -------
    List[int]
        an array, where for each vertex it is indicated
        how many triangles it participates in.
    """
    if not graph.square:
        raise ValueError("Adjacency matrix of the graph must be square")
    if graph.type != BOOL:
        raise ValueError(f"Unsupported graph type: {graph.type}. Expected type: BOOL")
    graph.union(graph.transpose(), out=graph)
    temp = graph.mxm(graph, cast=INT64, accum=INT64.PLUS, mask=graph)
    triangles_number = [sum(list(temp[i].vals)) // 2 for i in range(temp.nrows)]
    return triangles_number
