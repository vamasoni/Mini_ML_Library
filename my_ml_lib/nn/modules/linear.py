# my_ml_lib/nn/modules/linear.py
import numpy as np
from .base import Module
from my_ml_lib.nn.autograd import Value

class Linear(Module):
    """
    Applies a linear transformation to the incoming data: y = xW + b
    (Note: Implemented as x @ W + b where W is shape (in_features, out_features))
    """

    def __init__(self, in_features, out_features, bias=True):
        """
        Initializes the Linear layer.

        Args:
            in_features (int): Size of each input sample (number of input features).
            out_features (int): Size of each output sample (number of output features).
            bias (bool): If set to False, the layer will not learn an additive bias.
                         Default: True
        """
        # Initialize base class and store dimensions ---
        super().__init__() # CRITICAL: Initializes _parameters and _modules
        self.in_features = in_features
        self.out_features = out_features

        #  Initialize learnable parameters (Weight and Bias) ---
        # Initialize weight parameter as a Value object.
        # We use He initialization scaling, suitable for ReLU activations.
        # The weight matrix W has shape (in_features, out_features).

        scale = np.sqrt(2.0 / in_features)
        # The assignment self.weight = Value(...) automatically registers 'weight'
        # in self._parameters via the Module.__setattr__ method.
        self.weight = Value(scale * np.random.randn(in_features, out_features), label='weight')

        # Initialize bias parameter if requested.
        if bias:
            # Bias b has shape (out_features,). Initialized to zeros.
            # Assignment automatically registers 'bias' in self._parameters.
            self.bias = Value(np.zeros(out_features), label='bias')
        else:
            # If no bias, explicitly register None. This helps state_dict saving/loading.
            self.register_parameter('bias', None)

    def __call__(self, x: Value) -> Value:                    ### Can do this with or without broadcasting but modify accordingly###
        """
        Defines the forward pass of the Linear layer.

        Args:
            x (Value): Input Value object, expected shape (batch_size, in_features).

        Returns:
            Value: Output Value object, shape (batch_size, out_features).
        """
        # --- TODO: Implement the forward pass ---
        # Calculate the linear transformation: 
        # Remember to use the Value operations (__matmul__, __add__) to ensure
        # the computation graph is built correctly.

        # --- TODO: Step 1: Matrix Multiplication ---
        # Multiply the input 'x' by the layer's weight 'self.weight'.
        # Hint: Use the '@' operator (which calls Value.__matmul__).
        print("TODO: Implement matrix multiplication (x @ self.weight) in Linear.__call__")
        out = x # Placeholder

        # --- TODO: Step 2: Add Bias (Optional) ---
        # Check if a bias term exists (i.e., if self.bias is not None).
        # If it exists, add it to the result of the matrix multiplication.
        # Hint: Use the '+' operator (which calls Value.__add__).
        print("TODO: Implement bias addition (if self.bias exists) in Linear.__call__")
 

        return out # Return the final output Value object

    # String Representation ---
    def __repr__(self):
        """Provides a developer-friendly string representation of the layer."""
        # Check if bias parameter exists and is not None
        has_bias = self._parameters.get('bias', None) is not None
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={has_bias})"
