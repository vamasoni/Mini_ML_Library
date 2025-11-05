# scripts/visualize.py
"""
Visualize an autograd computation graph using Graphviz.
Usage: python scripts/visualize.py
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import graphviz
from utils.graph_utils import trace
from my_ml_lib.nn.autograd import Value
from my_ml_lib.nn.modules.linear import Linear
from my_ml_lib.nn.modules.activations import ReLU
from my_ml_lib.nn.modules.containers import Sequential
import numpy as np
from datetime import datetime
import os


def visualize(root, filename="graph"):
    """
    Build and render a Graphviz diagram of a computation graph starting from `root` Value.

    Parameters
    ----------
    root : Value
        Root node (typically a loss Value).
    filename : str
        Output file name (without extension).
    """
    nodes, edges = trace(root)
    dot = graphviz.Digraph(format="png", graph_attr={"rankdir": "LR"})

    # add nodes
    for n in nodes:
        uid = str(id(n))
        label = f"{n._op}|shape={n.data.shape}\nvalue={n.data}\ngrad={n.grad}"
        dot.node(name=uid, label=label, shape="record", style="filled", fillcolor="lightyellow")

    # add edges (parent → child)
    for parent, child in edges:
        dot.edge(str(id(parent)), str(id(child)))

    # save
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"results/{filename}_{timestamp}"
    dot.render(path, format="png", cleanup=True)
    print(f"[INFO] Graph saved to {path}.png")


def demo_tiny_mlp():
    """
    Demo: forward pass of a tiny MLP Sequential(Linear(2,3), ReLU()) on one sample.
    """
    np.random.seed(0)
    model = Sequential(Linear(2, 3), ReLU())
    x = np.random.randn(1, 2)
    v = Value(x)
    out = model(v)  # forward pass (Value graph built)
    print("[INFO] Forward pass output:", out.data)
    visualize(out, filename="tiny_mlp_graph")


if __name__ == "__main__":
    demo_tiny_mlp()
