import random
from abc import ABC, abstractmethod
from utilities import ExplanationType, ExplanationScope

import numpy as np


class Metric(ABC):
    """
    Base class for all metrics
    Extend this class and implement __call__ for custom metrics.
    """

    def __init__(self, dataset=None, explainer=None, seed=42, **kwargs):
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.dataset = dataset
        self.explainer = explainer

    @abstractmethod
    def __call__(self, **kwargs):
        """
        This is called to evaluate the metric.
        """
        raise NotImplementedError(
            'The scoring method of this test is not implemented at the moment.')

    def validate_explainer(self):
        """
        Validates if the explainers's explanation type and scope match the metric's explanation type and scope.

        Raises:
        ValueError: If the explainer's explanation type and scope do not match the metric's explanation type and scope.
        """
        explainer_explanation_type = getattr(
            self.explainer, 'explanation_type', None)
        explainer_scope = getattr(self.explainer, 'scope', None)

        metric_explanation_type = getattr(
            self, 'explanation_type', None)
        metric_scope = getattr(self, 'scope', None)

        explanation_type_match = (metric_explanation_type == explainer_explanation_type) or (
            metric_explanation_type == ExplanationType.ANY) | (explainer_explanation_type == ExplanationType.ANY)

        explanation_scope_match = (metric_scope == explainer_scope) or (
            metric_scope == ExplanationScope.ANY) | (explainer_scope == ExplanationScope.ANY)

        if not explanation_type_match or not explanation_scope_match:
            raise ValueError(f"Explainer type ({explainer_explanation_type}) and scope ({explainer_scope}) "
                             f"do not match the metric's expected explanation type ({metric_explanation_type}) "
                             f"and scope ({metric_scope}).")

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__class__.__name__
