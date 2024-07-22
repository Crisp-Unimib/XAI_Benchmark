import random
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Dataset(ABC):
    """
    Base class for all datasets
    Extend this class and implement for custom datasets.
    """

    def __init__(self, seed=42, n_features=None, **kwargs):
        """
        Initialize the dataset.

        Parameters
        ----------
        seed: int
            seed for reproducibility
        n_features: int
            number of features to use
        kwargs:
            additional arguments
        """
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.df = None
        self.X = None
        self.y = None
        self.y_predicted = None
        self.vectorizer = None
        self.model = None
        self.pipe = None

    @abstractmethod
    def __call__(self, **kwargs):
        """
        This is called to get the dataframe.

        Parameters
        ----------
        kwargs:
            additional arguments
        """
        raise NotImplementedError(
            'This dataset is not implemented at the moment.')

    def set_seed(self, seed):
        """
        Set the seed for random number generation.

        Parameters:
            seed (int): The seed value to set.
        """
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__class__.__name__
