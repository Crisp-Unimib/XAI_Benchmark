from .Metric import Metric
import logging
import numpy as np
from utilities import ExplanationType, ExplanationScope, MetricCategory

logger = logging.getLogger(__name__)


class NumberRulesMetric(Metric):
    """
    Metric to calculate the number of rules for a given dataset and explainer.

    Args:
        dataset (Dataset): The dataset to calculate the number of rules for.
        explainer (Explainer): The explainer object used to generate the rules.
        **kwargs: Additional keyword arguments.

    Attributes:
        scope (ExplanationScope): The scope of the explanation.
        explanation_type (ExplanationType): The type of explanation.
        metric_category (MetricCategory): The category of the metric.

    Raises:
        TypeError: If the explainer does not return a list of rules.
    """

    def __init__(self, dataset, explainer, **kwargs):
        """
        Initializes a new instance of the NumberRulesMetric class.

        Args:
            dataset: The dataset used for evaluation.
            explainer: The explainer used for generating explanations.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(dataset, explainer, **kwargs)
        self.scope = ExplanationScope.ANY
        self.explanation_type = ExplanationType.RULE
        self.metric_category = MetricCategory.PARSIMONY
        self.validate_explainer()

    def __call__(self, instance=None):
        """
        Calculates the number of rules for the given dataset.

        Returns:
            int: The count of rules.

        Raises:
            TypeError: If the explainer does not return a list of rules.
        """
        if self.explainer.scope.value == 'local':
            if not instance:
                instance = self.dataset.X[0]
            rules = self.explainer(instance, n_rules=1000)
        else:
            rules = self.explainer(n_rules=1000)
        if not isinstance(rules, list):
            raise TypeError("Expected a list of rules from the explainer")
        # The metric is the count of rules
        number_of_rules = len(rules)
        logger.info(
            f"Number of rules for the given dataset: {number_of_rules}")
        return number_of_rules
