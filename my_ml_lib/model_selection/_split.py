# my_ml_lib/model_selection/_split.py
import numpy as np

def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    """
    Split X and y arrays into random train and test subsets.

    Args:
        X (array-like): Feature data, shape (n_samples, n_features).
        y (array-like): Target labels, shape (n_samples,).
        test_size (float or int): Proportion (0.0-1.0) or absolute number for the test split.
        shuffle (bool): Whether to shuffle data before splitting.
        random_state (int, optional): Seed for shuffling reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    n_samples = X.shape[0]
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    # --- TODO: Step 1 - Calculate n_test and n_train ---
    # Check if test_size is float or int and calculate n_test.
    # Handle edge cases (invalid values for test_size).
    # n_train = n_samples - n_test.
    print("TODO: Implement calculation of n_test and n_train in train_test_split.")


    # --- TODO: Step 2 - Create and Shuffle Indices ---
    # indices = np.arange(n_samples)
    # If shuffle is True:
    #   Initialize a RandomState generator: use np.random.RandomState(random_state)
    #   Shuffle indices in place:
    print("TODO: Implement index creation and shuffling in train_test_split.")
    

    # --- TODO: Step 3 - Split Indices ---into train_indices and test_indices
  
    print("TODO: Implement index splitting in train_test_split.")
 

    # --- TODO: Step 4 - Split Arrays ---
    # eg X_train = X[train_indices]
  
    print("TODO: Implement array splitting using indices in train_test_split.")
  

    return X_train, X_test, y_train, y_test


def train_test_val_split(X, y,
                         train_size=0.7,
                         val_size=0.15,
                         test_size=0.15,
                         shuffle=True,
                         random_state=None):
    """
    Split X and y arrays into random train, validation, and test subsets.

    Args:
        X (array-like): Feature data, shape (n_samples, n_features).
        y (array-like): Target labels, shape (n_samples,).
        train_size (float): Proportion for the train split (0.0 to 1.0).
        val_size (float): Proportion for the validation split (0.0 to 1.0).
        test_size (float): Proportion for the test split (0.0 to 1.0). Must sum to 1.0.
        shuffle (bool): Whether to shuffle data before splitting.
        random_state (int, optional): Seed for shuffling reproducibility.

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    n_samples = X.shape[0]
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    # --- TODO: Step 1 - Validate Proportions ---
    # Check if train_size, val_size, test_size are valid floats between 0 and 1.
    # Check if they sum (approximately, use np.isclose) to 1.0. Raise ValueError if not.
    print("TODO: Implement proportion validation in train_test_val_split.")

    # --- TODO: Step 2 - Calculate Split Sizes ---
    # Calculate n_train, n_val using np.floor or similar.
    # Calculate n_test = n_samples - n_train - n_val to ensure all samples are used.
    # Check if any calculated size is 0 and raise ValueError if so.
    print("TODO: Implement calculation of n_train, n_val, n_test in train_test_val_split.")
   

    # --- TODO: Step 3 - Create and Shuffle Indices ---
    # indices = np.arange(n_samples)
    # If shuffle is True, initialize RandomState and shuffle indices.
    print("TODO: Implement index creation and shuffling in train_test_val_split.")

    # --- TODO: Step 4 - Split Indices ---
    # Determine split points based on n_train and n_val.
    # eg train_indices = indices[:n_train]

    print("TODO: Implement index splitting in train_test_val_split.")


    # --- TODO: Step 5 - Split Arrays ---
    # eg  X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]

    print("TODO: Implement array splitting using indices in train_test_val_split.")
\
    return X_train, X_val, X_test, y_train, y_val, y_test