
import numpy as np 


class KFold:
    """
    K-Folds cross-validator.

    Provides train/test indices to split data in train/test sets.
    Splits dataset into k consecutive folds (without shuffling by default).
    Each fold is then used once as a validation set while the k - 1 remaining
    folds form the training set.
    """
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        """
        Initializes the KFold splitter.

        Args:
            n_splits (int): Number of folds. Must be at least 2.
            shuffle (bool): Whether to shuffle the data before splitting into batches.
                            Note that shuffling is done each time split() is called.
            random_state (int, optional): When shuffle is True, random_state affects the ordering
                                           of the indices. Pass an int for reproducible output.
        """
        if not isinstance(n_splits, int) or n_splits <= 1:
            raise ValueError("n_splits must be an integer greater than 1.")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a boolean value.")

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Args:
            X (array-like): Data to split, shape (n_samples, n_features).
            y (array-like, optional): The target variable for supervised learning problems.
                                      Used for stratification in StratifiedKFold, but not here.
            groups (array-like, optional): Group labels for the samples used while splitting the dataset
                                            into train/test set. Not used in standard KFold.

        Yields:
            tuple: (train_indices, test_indices) arrays for each split.
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        # TODO: Implement shuffling logic if self.shuffle is True
        if self.shuffle:
            print("KFold: Shuffling needs implementation using self.random_state.")


        # TODO: Implement the logic to calculate fold sizes and yield indices
        print("KFold: Splitting logic needs implementation.")
        
        
        # Determine fold sizes, distributing remainder samples across first folds 
        # Hint  : Can use np.full  and then redestribute accordingly



        current = 0
        # yield to make it a generator
        for fold_size in fold_sizes:
            # Define start and end indices for the current test fold 

            # Select test indices for this fold

             # Select train indices (all indices *except* the test ones) (Hint : Use Masking Logic )

            # yield train_indices, test_indices
            yield (np.array([]), np.array([]))

            # Move to the start of the next fold

            current = ?

             

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits
    

