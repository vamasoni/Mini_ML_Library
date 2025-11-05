class Sequential:
    def __init__(self, *layers):
        self.layers = list(layers)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        out = x
        for l in self.layers:
            out = l(out)
        return out

    def parameters(self):
        for l in self.layers:
            if hasattr(l, 'parameters'):
                for name, p in l.parameters():
                    yield f"{l.__class__.__name__}.{name}", p
