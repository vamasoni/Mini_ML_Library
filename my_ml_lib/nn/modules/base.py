# my_ml_lib/nn/modules/base.py
from my_ml_lib.nn.autograd import Value # Import Value from our autograd engine
import numpy as np
from collections import OrderedDict # Use OrderedDict

class Module:
    """Base class for all neural network modules (layers, containers, etc.)."""

    def __init__(self):
        """Initializes internal dictionaries for parameters and submodules."""
        # These dictionaries will store the learnable parameters (Value objects)
        # and sub-modules (other Module objects) contained within this module.
        # Use OrderedDict to keep insertion order (useful for sequential, etc.)


        self._parameters = OrderedDict()
        self._modules = OrderedDict()

    #  Registration Helpers ---
    # These methods are used internally (often via __setattr__) to correctly
    # register parameters and sub-modules so they can be discovered later.

    def register_parameter(self, name: str, param):
        """Adds a parameter (Value object) to the module."""
        if not isinstance(param, Value) and param is not None:
             raise TypeError(f"cannot assign {type(param)} as parameter '{name}' "
                             "(Value or None expected)")
        if '.' in name: raise KeyError("parameter name can't contain \".\"")
        if name == '': raise KeyError("parameter name can't be empty string \"\"")
        self._parameters[name] = param

    def add_module(self, name: str, module):
        """Adds a child module (another Module object) to the current module."""
        if not isinstance(module, Module) and module is not None:
             raise TypeError(f"{module} is not a Module subclass")
        if '.' in name: raise KeyError("module name can't contain \".\"")
        if name == '': raise KeyError("module name can't be empty string \"\"")
        self._modules[name] = module

    #  Parameter Discovery ---
    # This helper function recursively finds all parameters within this module
    # and its submodules, assigning them hierarchical names (e.g., 'layer1.weight').

    def _get_named_parameters(self, prefix=''):
        """Helper to recursively get named parameters."""
        memo = set()# Keep track of parameter objects already yielded
        # Yield parameters directly held by this module
        for name, param in self._parameters.items():
            if param is not None and param not in memo:
                memo.add(param)
                yield prefix + ('.' if prefix else '') + name, param

        # Recursively yield parameters from submodules

        for name, module in self._modules.items():
            if module is not None:
                    # The recursive call handles adding the submodule name to the prefix

                yield from module._get_named_parameters(prefix + ('.' if prefix else '') + name)

    # --- TODO: Implement Parameter Access ---
    def parameters(self):
        """
        Return an iterable (e.g., a list or generator) of all Value parameters
        in this module and its submodules.

        Hint: Use the `_get_named_parameters` helper method. You only need to
              yield or collect the parameter `Value` objects, not their names.
        """
        print("TODO: Implement the parameters() method.")
        # Example: yield from (param for name, param in self._get_named_parameters())
        # Or: return list(param for name, param in self._get_named_parameters())
        yield Value(np.array(0.0)) # Placeholder




        # --- TODO: Implement Gradient Zeroing ---
    def zero_grad(self):
        """Sets gradients (.grad attribute) of all parameters to zero."""

        print("TODO: Implement gradient zeroing loop in zero_grad().")
        pass # Placeholder

    #  Forward Pass Placeholder ---
    def __call__(self, *args, **kwargs):
        """Defines the forward pass. Must be implemented by subclasses."""
        # Subclasses (like Linear, ReLU, Sequential, or custom models) MUST override this method.
        raise NotImplementedError("Subclasses must implement the forward pass (__call__)")

    




############################# Saving and Loading Logic #################################
## Basically we are storing state_dictionary which stores the weights and biases as dictionary 

### For easiness, its given to you all to take reference from.

    #  State Dictionary Retrieval ---
    # This uses the parameter discovery mechanism to create a dictionary
    # mapping parameter names to their actual data (NumPy arrays).

    def state_dict(self):
        """Returns a dictionary containing the module's state (parameter data)."""
        return {name: param.data for name, param in self._get_named_parameters()}

    def save_state_dict(self, filepath):
        """Saves the module's state dictionary to a .npz file."""
        # --- STUDENT IMPLEMENTATION (Correct as provided) ---
        current_state_dict = self.state_dict()
        try:
            np.savez_compressed(filepath, **current_state_dict)
            print(f"State dictionary saved to {filepath}")
        except Exception as e:
            print(f"Error saving state dictionary: {e}")

    #  State Dictionary Loading Given to you all !!!
    def load_state_dict(self, filepath):
        """Loads the module's state dictionary from a .npz file."""
        try:
            loaded_state_dict_npz = np.load(filepath)
            # Convert NpzFile to a regular dict for easier key checking
            loaded_state_dict = {k: loaded_state_dict_npz[k] for k in loaded_state_dict_npz.files}
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return
        except Exception as e:
            print(f"Error loading state dictionary: {e}")
            return

        current_params_dict = dict(self._get_named_parameters())
        print(f"Loading state dictionary from {filepath}...")
        loaded_keys = set(loaded_state_dict.keys())
        model_keys = set(current_params_dict.keys())

        missing_keys = model_keys - loaded_keys
        unexpected_keys = loaded_keys - model_keys
        if missing_keys: print(f"Warning: Missing keys in state_dict: {missing_keys}")
        if unexpected_keys: print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")

        # Assign values
        for name, param in current_params_dict.items():
            if name in loaded_state_dict:
                loaded_array = loaded_state_dict[name]
                if param.data.shape != loaded_array.shape:
                    print(f"Error: Shape mismatch for parameter '{name}'. "
                          f"Model has {param.data.shape}, loaded has {loaded_array.shape}.")
                    continue # Skip assignment on shape mismatch

                # Assign data IN PLACE
                param.data[:] = loaded_array
            # else: # Handled by missing_keys warning

        print("State dictionary loaded successfully (check warnings/errors above).")







    # Automatic Registeration or parameters

    # This __setattr__ automatically registers Value objects as parameters
    # and Module objects as submodules when you assign them as attributes

    # This is the magic part! It automatically calls register/add  (Given to you all!!!!)
    def __setattr__(self, name, value):
         """Override setattr to automatically register Modules and Parameters."""

         # --- Condition 1: Handle internal attributes or non-Module/non-Value ---
         # If name starts with _, treat as internal - just assign normally.
         # Also, if value isn't a Module or Value, just assign normally.
         if name.startswith('_') or not (isinstance(value, Value) or isinstance(value, Module)):
             super().__setattr__(name, value)
             return # Skip registration logic

         # --- Condition 2: Register Parameter (Value) ---
         if isinstance(value, Value):
             # Ensure _parameters dict exists
             if '_parameters' not in self.__dict__:
                 raise AttributeError("cannot assign parameter before Module.__init__() call")
             # Remove existing attribute if it was previously registered differently
             # (e.g., changing a submodule to a parameter)
             self._modules.pop(name, None)
             # Register/Update the parameter
             self.register_parameter(name, value) # Use the registration method

         # --- Condition 3: Register Submodule (Module) ---
         elif isinstance(value, Module):
             # Ensure _modules dict exists
             if '_modules' not in self.__dict__:
                 raise AttributeError("cannot assign module before Module.__init__() call")
             # Remove existing attribute if it was previously registered differently
             self._parameters.pop(name, None)
             # Register/Update the submodule
             self.add_module(name, value) # Use the registration method

         # --- Always store the attribute directly using super() ---
         # This ensures self.weight, self.layer_one etc. are accessible
         super().__setattr__(name, value)