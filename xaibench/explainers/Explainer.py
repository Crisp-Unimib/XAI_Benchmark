import random
from abc import ABC, abstractmethod

import numpy as np


class Explainer(ABC):
    """
    Base class for all evaluators
    Extend this class and implement __call__ for custom evaluators.
    """

    def __init__(self, seed=42, **kwargs):
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    @abstractmethod
    def __call__(self, model):
        """
        This is called to evaluate the model.

        Parameters
        ----------
        model:
            the model to evaluate
        """
        raise NotImplementedError(
            'The explaining method of this explainer is not implemented at the moment.')

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
