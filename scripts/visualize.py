# scripts/visualize.py
"""
Visualize an autograd computation graph using Graphviz.

Usage examples:
  # default tiny model (input 2 -> hidden [3] with ReLU)
  python -m scripts.visualize

  # custom model, input dim 3, hidden layers 5 and 4, activation tanh
  python -m scripts.visualize --input_dim 3 --hidden 5 4 --activation tanh

  # choose leakyrelu and save to custom base filename
  python -m scripts.visualize --hidden 8 --activation leakyrelu --outfile mygraph
"""
import sys
import os
import glob
import shutil
import argparse
from datetime import datetime

# ensure project root is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ---------------------------------------------------------------------
# Try to auto-add Graphviz to PATH on Windows if dot not found
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

# now import graphviz (will fail if Python package not installed)
try:
    import graphviz
except Exception as e:
    raise RuntimeError("Please install the Python 'graphviz' package (pip install graphviz).") from e

# imports from your library
from utils.graph_utils import trace
from my_ml_lib.nn.autograd import Value
from my_ml_lib.nn.modules.linear import Linear
from my_ml_lib.nn.modules.containers import Sequential

# Try to import activation modules; provide fallbacks if not present
try:
    from my_ml_lib.nn.modules.activations import ReLU, Sigmoid, Tanh, LeakyReLU, GELU
    _HAS_ACT_MODULES = True
except Exception:
    _HAS_ACT_MODULES = False
    # Define simple fallback activation wrappers that behave like Modules (callable)
    class ReLU:
        def __call__(self, x: Value):
            return x.relu()
        def __repr__(self):
            return "ReLU()"

    class Sigmoid:
        def __call__(self, x: Value):
            return x.sigmoid()
        def __repr__(self):
            return "Sigmoid()"

    class Tanh:
        def __call__(self, x: Value):
            # implement via numpy tanh on Value.data -> wrap to Value to keep graph connection
            out = Value(np.tanh(x.data), _children=(x,), op="tanh")
            def _backward():
                g = out.grad
                # derivative: 1 - tanh^2(x)
                grad_mask = 1.0 - (out.data ** 2)
                x.grad = x.grad + x._unbroadcast(g * grad_mask, x.data.shape)
            out._backward = _backward
            return out
        def __repr__(self):
            return "Tanh()"

    class LeakyReLU:
        def __init__(self, negative_slope=0.01):
            self.negative_slope = negative_slope
        def __call__(self, x: Value):
            s = self.negative_slope
            out = Value(np.where(x.data > 0, x.data, s * x.data), _children=(x,), op=f"leakyrelu({s})")
            def _backward():
                g = out.grad
                mask = np.where(x.data > 0, 1.0, s)
                x.grad = x.grad + x._unbroadcast(g * mask, x.data.shape)
            out._backward = _backward
            return out
        def __repr__(self):
            return f"LeakyReLU({self.negative_slope})"

    class GELU:
        def __call__(self, x: Value):
            # approximate GELU
            a = x.data
            c = (2.0 / np.pi) ** 0.5
            inner = c * (a + 0.044715 * (a**3))
            out_data = 0.5 * a * (1 + np.tanh(inner))
            out = Value(out_data, _children=(x,), op="gelu")
            def _backward():
                g = out.grad
                tanh_term = np.tanh(inner)
                d = 0.5 * (1 + tanh_term) + 0.5 * a * (1 - tanh_term**2) * c * (1 + 3 * 0.044715 * a**2)
                x.grad = x.grad + x._unbroadcast(g * d, x.data.shape)
            out._backward = _backward
            return out
        def __repr__(self):
            return "GELU()"

# numpy import for fallback ops & array formatting
import numpy as np

# ---------------------------------------------------------------------
# Graph visualization utilities
# ---------------------------------------------------------------------
def visualize(root: Value, filename: str = "graph"):
    """
    Render a computation graph starting from `root` Value into results/{filename}_{timestamp}.png
    """
    nodes, edges = trace(root)
    dot = graphviz.Digraph(format="png", graph_attr={"rankdir": "LR"})

    # add nodes with labels including op name and shape and small preview of value/grad
    for n in nodes:
        uid = str(id(n))
        val_str = np.array2string(n.data, precision=3, threshold=6, edgeitems=2)
        grad_str = np.array2string(n.grad, precision=3, threshold=6, edgeitems=2)
        label = f"{n._op}|shape={n.data.shape}\\nval={val_str}\\ngrad={grad_str}"
        dot.node(name=uid, label=label, shape="record", style="filled", fillcolor="lightyellow")

    # add edges parent -> child
    for parent, child in edges:
        dot.edge(str(id(parent)), str(id(child)))

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"results/{filename}_{timestamp}"
    # render and cleanup intermediate files
    dot.render(path, format="png", cleanup=True)
    print(f"[INFO] Graph saved to {path}.png")


# ---------------------------------------------------------------------
# Build model helper
# ---------------------------------------------------------------------
def build_sequential(input_dim: int, hidden_sizes: list, activation_name: str):
    """
    Build a Sequential model where for each hidden size we add Linear(prev, h) followed by activation.
    activation_name: 'relu','sigmoid','tanh','leakyrelu','gelu'
    """
    # activation instance mapping
    act_name = activation_name.lower()
    if act_name == "relu":
        act_inst = ReLU()
    elif act_name == "sigmoid":
        act_inst = Sigmoid()
    elif act_name == "tanh":
        act_inst = Tanh()
    elif act_name == "leakyrelu":
        act_inst = LeakyReLU()
    elif act_name == "gelu":
        act_inst = GELU()
    else:
        raise ValueError(f"Unknown activation '{activation_name}'")

    layers = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(Linear(prev, h))
        layers.append(act_inst)
        prev = h
    # If you want final linear output layer (optional), keep as-is (here we end at last activation).
    model = Sequential(*layers)
    return model

# ---------------------------------------------------------------------
# Demo-run CLI entrypoint
# ---------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Visualize autograd computation graph for a tiny MLP.")
    parser.add_argument("--input_dim", type=int, default=2, help="Input feature dimensionality")
    parser.add_argument("--hidden", type=int, nargs="+", default=[3], help="Hidden layer sizes (space-separated)")
    parser.add_argument("--activation", type=str, default="relu", help="Activation: relu|sigmoid|tanh|leakyrelu|gelu")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--outfile", type=str, default=None, help="Base name for output file (placed in results/). If omitted, automatic name is used.")
    args = parser.parse_args(argv)

    np.random.seed(args.seed)
    print(f"[INFO] Building model: input_dim={args.input_dim}, hidden={args.hidden}, activation={args.activation}")

    model = build_sequential(args.input_dim, args.hidden, args.activation)
    print(f"[INFO] Model built: {model}")

    # single sample input
    x = np.random.randn(1, args.input_dim)
    v = Value(x)
    out = model(v)
    print("[INFO] Forward pass output shape:", out.data.shape)
    # choose filename
    if args.outfile:
        fname = args.outfile
    else:
        fname = f"mlp_{args.activation}_{args.input_dim}_{'_'.join(map(str,args.hidden))}"
    visualize(out, filename=fname)

if __name__ == "__main__":
    main()


#defaut run
# python -m scripts.visualize

#custom activation and hidden layers
# python -m scripts.visualize --input_dim 3 --hidden 5 4 --activation tanh

#custom filename
# python -m scripts.visualize --outfile mygraph