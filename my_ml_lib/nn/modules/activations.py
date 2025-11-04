# my_ml_lib/nn/modules/activations.py
class ReLU:
    def __call__(self, x):
        return x.relu()

class Sigmoid:
    def __call__(self, x):
        # implement sigmoid via Value nodes: sigmoid(x) = 1/(1+exp(-x))
        return (1 + (-x).exp()) ** -1
