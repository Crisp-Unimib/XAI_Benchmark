from .Metric import Metric
import logging
import numpy as np
from utilities import ExplanationType, ExplanationScope, MetricCategory

logger = logging.getLogger(__name__)


class AvgRuleLengthMetric(Metric):
    """
    Metric to calculate the average rule length for a given dataset and explainer.

    Args:
        dataset (Dataset): The dataset to calculate the rules length for.
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
        Calculates the average length of rules generated by the explainer.

        Returns:
            float: The average length of rules.

        Raises:
            TypeError: If the explainer does not return a list of rules.
        """
        def avg_length(rules):
            total_length = sum(len(rule) for rule in rules)
            average_length = total_length / len(rules)
            return average_length

        # Retrieve rules from the explainer
        if self.explainer.scope.value == 'global':
            rules = self.explainer()
        else:
            if instance is None:
                instance = self.dataset.X[0:10]
            # Local case: we have a list of explanations
            rules = [self.explainer(i) for i in instance]
            rules = [avg_length(rule) for rule in rules]
        if not isinstance(rules, list):
            raise TypeError(
                "The explainer must return a list of rules.")

        # Calculate the average rule length
        average_length = np.average(rules)
        return average_length
