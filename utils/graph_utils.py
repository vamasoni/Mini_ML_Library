# my_ml_lib/utils/graph_utils.py
"""
Utilities for visualizing the autograd computation graph.
"""

def trace(root):
    """
    Recursively collect all nodes and edges in the computation graph
    starting from the given root Value.

    Returns
    -------
    nodes : set of Value
    edges : set of (parent, child) tuples
    """
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for parent in getattr(v, "_prev", []):
                edges.add((parent, v))
                build(parent)

    build(root)
    return nodes, edges
