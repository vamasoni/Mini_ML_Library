# my_ml_lib/nn/autograd.py
import numpy as np
import math # For exp, log etc.






class Value:
    """
    Stores a scalar or numpy array and its gradient.
    Builds a computation graph for automatic differentiation (backpropagation).
    Inspired by micrograd: https://github.com/karpathy/micrograd
    """
    def __init__(self, data, _parents=(), _op='', label=''):
        """
        Initializes a Value object.

        Args:
            data: Input data (convertible to numpy float64 array).
            _parents (tuple): Parent Value objects that created this node.
            _op (str): Operation that created this node.
            label (str): Optional label for debugging/visualization.
        """
        # Data Type Conversion & Initialization ---
        # Ensure data is a numpy array for consistency later
        if not isinstance(data, np.ndarray):
            # Try converting lists/tuples/scalars to numpy arrays
            try:
                data = np.array(data, dtype=np.float64)
            except TypeError:
                raise TypeError(f"Data must be convertible to a numpy array, got {type(data)}")
        # Ensure data is float64 for precision
        if not np.issubdtype(data.dtype, np.floating):
             print(f"Warning: Casting data from {data.dtype} to float64.")
             data = data.astype(np.float64)

        self.data = data
        # Initialize gradient with zeros, matching data shape
        self.grad = np.zeros_like(data, dtype=np.float64)
        # Internal variables for graph structure
        self._backward = lambda: None # Function to compute local gradients
        self._prev = set(_parents)   # Set of parent Value objects
        self._op = _op                # Operation that created this Value
        self.label = label            # Optional label for debugging/visualization


    def __repr__(self):
         # Provide a nice representation, showing shape for arrays
        data_str = f"array(shape={self.data.shape})" if self.data.ndim > 0 else f"scalar({self.data.item():.4f})"
        grad_str = f"array(shape={self.grad.shape})" if self.grad.ndim > 0 else f"scalar({self.grad.item():.4f})"
        return f"Value(data={data_str}, grad={grad_str}, op='{self._op}')"

    # --- TODO: Implement Core Operations ---
    ## Add operation with broadcasting has been given as an example 

    def __add__(self, other):
        # Ensure 'other' is also a Value object, wrap if necessary
        other = other if isinstance(other, Value) else Value(other)
        # Perform addition using numpy's broadcasting rules
        out_data = self.data + other.data
        out = Value(out_data, (self, other), '+')

        def _backward():
            # Gradient of '+' wrt self is 1, wrt other is 1.
            # Chain rule: self.grad += d(out)/d(self) * out.grad = 1 * out.grad
            # Need to handle broadcasting correctly for gradients!
            # If shapes were broadcasted, sum the gradient to match input shape.

            # Calculate gradient contribution
            grad_self = out.grad
            grad_other = out.grad    

            # Undo broadcasting for self.grad
            if self.data.shape != grad_self.shape:
                axis_to_sum = tuple(range(grad_self.ndim - self.data.ndim))
                grad_self = np.sum(grad_self, axis=axis_to_sum)
                if grad_self.shape != self.data.shape: # Handle singleton dimensions
                     dims_to_squeeze = tuple(i for i, dim in enumerate(self.data.shape) if dim == 1)
                     grad_self = np.sum(grad_self, axis=dims_to_squeeze, keepdims=True)

            # Undo broadcasting for other.grad
            if other.data.shape != grad_other.shape:
                # 1. Sum over the "new" axes (e.g., if (3,) + () -> (3,))
                axis_to_sum = tuple(range(grad_other.ndim - other.data.ndim))
                grad_other = np.sum(grad_other, axis=axis_to_sum)
                
                if grad_other.shape != other.data.shape: # Handle singleton dimensions
                     dims_to_squeeze = tuple(i for i, dim in enumerate(other.data.shape) if dim == 1)
                     grad_other = np.sum(grad_other, axis=dims_to_squeeze, keepdims=True)

            # Accumulate gradients
            self.grad += grad_self
            other.grad += grad_other

        out._backward = _backward
        return out

    def __mul__(self, other):
        """ Multiplication operation. Handles broadcasting. """
        other = other if isinstance(other, Value) else Value(other) # Ensure other is Value
        # --- TODO: Step 1 - Forward Pass ---
        # Calculate the output data: 
        print("TODO: Implement forward pass for __mul__")
        out_data = self.data # Placeholder
        out = Value(out_data, (self, other), '*')

        # --- TODO: Step 2 - Define _backward for Multiplication ---
        def _backward():
            # Gradient of 'A * B' w.r.t A is B.data, w.r.t B is A.data.
            # Calculate the gradient contributution
            # IMPORTANT: Handle broadcasting similar to __add__._backward. Sum gradients appropriately.
            print("TODO: Implement _backward for __mul__, including broadcasting if desired ")

            # Perform necessary np.sum operations on grad_self/grad_other if shapes mismatch
            
            pass # Placeholder

        out._backward = _backward
        return out
    
    # --- Make addition and multiplication commutative ---
    def __radd__(self, other): # other + self
        return self + other

    def __rmul__(self, other): # other * self
        return self * other

    # --- Other necessary math operations ---
    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __truediv__(self, other): # self / other
        # Division is multiplication by the inverse (power of -1)
        return self * (other**-1)

    def __rtruediv__(self, other): # other / self
        return other * (self**-1)
    





    def __pow__(self, other):
        """ Power operation (only supports scalar power). """
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now"
        # --- TODO: Step 1 - Forward Pass ---
        print("TODO: Implement forward pass for __pow__")
        out_data = self.data # Placeholder
        out = Value(out_data, (self,), f'**{other}')

        # --- TODO: Step 2 - Define _backward for Power ---
        def _backward():
           
            print("TODO: Implement _backward for __pow__.")
            pass # Placeholder

        out._backward = _backward
        return out

    # --- TODO: Implement Activation Functions 
    def relu(self):
        """ Rectified Linear Unit (ReLU) activation. """
        # --- TODO: Step 1 - Forward Pass ---
        # Calculate output data: 
        print("TODO: Implement forward pass for relu.")
        out_data = self.data # Placeholder
        out = Value(out_data, (self,), 'ReLU')

        # --- TODO: Step 2 - Define _backward for ReLU ---
        def _backward():
     
            print("TODO: Implement _backward for relu.")
            pass # Placeholder

        out._backward = _backward
        return out

    # --- TODO: Implement Elementary Functions (exp, log) ---
    def exp(self):
        """ Exponential function. """
        # --- TODO: Step 1 - Forward Pass ---
        # Calculate output data: 
        # Consider clipping self.data to avoid overflow (e.g., np.clip(self.data, -500, 700))
        print("TODO: Implement forward pass for exp (with clipping).")
        out_data = self.data # Placeholder
        out = Value(out_data, (self,), 'exp')

        # --- TODO: Step 2 - Define _backward for exp ---
        def _backward():
        
            print("TODO: Implement _backward for exp.")
            pass # Placeholder

        out._backward = _backward
        return out

    def log(self):
        """ Natural logarithm function (log base e). """
        # --- TODO: Step 1 - Forward Pass ---
        # Calculate output data: 
        # IMPORTANT: Ensure numerical stability. Avoid log(0) or log(negative).
        # Hint: Use np.maximum(self.data, epsilon) where epsilon is small (e.g., 1e-15).
        print("TODO: Implement forward pass for log (with stability).")
        out_data = self.data # Placeholder
        out = Value(out_data, (self,), 'log')

        # --- TODO: Step 2 - Define _backward for log ---
        def _backward():
    
            # IMPORTANT: Ensure numerical stability here too (avoid division by zero).
            # Hint: Use the same stable data as in the forward pass for the division.
            print("TODO: Implement _backward for log (with stability).")
      
            pass # Placeholder

        out._backward = _backward
        return out

    # --- TODO: Implement Matrix Multiplication ---
    def __matmul__(self, other):
        """ Matrix multiplication (@ operator). """
        other = other if isinstance(other, Value) else Value(other) # Ensure other is Value
        # --- TODO: Step 1 - Forward Pass ---
        # Calculate output data: 
        print("TODO: Implement forward pass for __matmul__.")
        out_data = self.data # Placeholder
        out = Value(out_data, (self, other), '@')

        # --- TODO: Step 2 - Define _backward for matmul ---
        def _backward():
           
            # IMPORTANT: Handle shapes and transposes correctly (np.transpose() or .T).
            # Consider potential batch dimensions (although simple MLP might not need complex handling).

            print("TODO: Implement _backward for __matmul__.")

            # Ensure gradients are initialized

            # Gradient for self (A)

            # If batch dimensions were involved, sum gradients over them

            # Handle potential broadcasting dims that need summing (Your choice  )


            pass # Placeholder

        out._backward = _backward
        return out

    # --- TODO: Implement Reduction Operations (sum, mean) ---
    def sum(self, axis=None, keepdims=False):
        """ Summation operation. """
        # --- TODO: Step 1 - Forward Pass ---
        # Calculate output data:
        print("TODO: Implement forward pass for sum.")
        out_data = self.data # Placeholder
        out = Value(out_data, (self,), 'sum')

        # --- TODO: Step 2 - Define _backward for sum ---
        def _backward():
            # Gradient of sum is 1, distributed back to inputs.
            # If summed over specific axes, the gradient needs to be broadcast back correctly.
            # Hint: Use np.ones_like(self.data) multiplied by out.grad (potentially reshaped).
            print("TODO: Implement _backward for sum (handle axis/keepdims).")
   
            pass # Placeholder

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        """ Mean operation. """
        # --- TODO: Step 1 - Forward Pass ---
        # Calculate output data: 
        print("TODO: Implement forward pass for mean.")
        out_data = self.data # Placeholder
        out = Value(out_data, (self,), 'mean')

        # --- TODO: Step 2 - Define _backward for mean ---
        def _backward():
            # Gradient of mean is 1/N, distributed back. N is the number of elements averaged over.
            # Calculate N based on self.data.shape and the axis argument.
            # Distribute out.grad / N back to self.grad, handling broadcasting.
            print("TODO: Implement _backward for mean (calculate N, handle axis/keepdims).")
        
            pass # Placeholder

        out._backward = _backward
        return out

    #  BACKPROPAGATION 
    def backward(self):
        """ Performs backpropagation starting from this Value node. """
        # --- TODO: Step 1 - Topological Sort ---
        # Build a list `topo` containing all nodes in the graph leading to this node,
        # in topologically sorted order (parents before children).
        # Use a `visited` set to avoid infinite loops in cyclic graphs (shouldn't happen here).
        # Hint: Implement a recursive helper function `build_topo(v)`.



        # --- TODO: Step 2 - Initialize Gradient ---
        # Initialize gradient of the final node (self) to ones (or appropriate shape)

    
        print("TODO: Initialize self.grad in backward().")


        # --- TODO: Step 3 - Backward Pass ---
        # Iterate through the `topo` list in reverse order.
        # For each node, call its `_backward()` method. This applies the chain rule locally.

        print("TODO: Implement the backward pass loop in backward().")

        pass # Placeholder

