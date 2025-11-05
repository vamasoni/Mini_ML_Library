# scripts/visualize.py
"""
Visualize an autograd computation graph using Graphviz.

Usage:
    python -m scripts.visualize
"""

import sys, os, glob, shutil
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ---------------------------------------------------------------------
# ✅ AUTO Graphviz PATH FIX for Windows
# ---------------------------------------------------------------------
if shutil.which("dot") is None:
    common_paths = (
        glob.glob(r"C:\Program Files*\Graphviz*\bin") +
        glob.glob(r"C:\Program Files (x86)\Graphviz*\bin")
    )
    for p in common_paths:
        if os.path.exists(os.path.join(p, "dot.exe")):
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
            print(f"[INFO] Added Graphviz to PATH temporarily: {p}")
            break

import graphviz
from utils.graph_utils import trace
from my_ml_lib.nn.autograd import Value
from my_ml_lib.nn.modules.linear import Linear
from my_ml_lib.nn.modules.activations import ReLU
from my_ml_lib.nn.modules.containers import Sequential
import numpy as np

# ---------------------------------------------------------------------
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
        val_str = np.array2string(n.data, precision=3, threshold=4, edgeitems=2)
        grad_str = np.array2string(n.grad, precision=3, threshold=4, edgeitems=2)
        label = f"{n._op}|shape={n.data.shape}\\nval={val_str}\\ngrad={grad_str}"
        dot.node(name=uid, label=label, shape="record", style="filled", fillcolor="lightyellow")

    # add edges (parent → child)
    for parent, child in edges:
        dot.edge(str(id(parent)), str(id(child)))

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"results/{filename}_{timestamp}"
    dot.render(path, format="png", cleanup=True)
    print(f"[INFO] Graph saved to {path}.png")


# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
if __name__ == "__main__":
    demo_tiny_mlp()
