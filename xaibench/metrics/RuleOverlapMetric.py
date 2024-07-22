from .Metric import Metric
import logging
from itertools import combinations
from utilities import ExplanationType, ExplanationScope, MetricCategory
import numpy as np

logger = logging.getLogger(__name__)


class RuleOverlapMetric(Metric):
    def __init__(self, dataset, explainer, **kwargs):
        """
        Initializes a RuleOverlapMetric object.

        Args:
            dataset: The dataset used for evaluation.
            explainer: The explainer used for generating explanations.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        super().__init__(dataset, explainer, **kwargs)
        self.scope = ExplanationScope.ANY
        self.explanation_type = ExplanationType.RULE
        self.metric_category = MetricCategory.CONTEXTFULNESS
        self.validate_explainer()

    def __call__(self, instance=None, n_rules=10):
        """
        Calculates the fraction overlap between rules in the explainer.

        Parameters:
            instance (optional): The instance for which the rules are calculated. If not provided, the rules are calculated for the entire dataset.

        Returns:
            The fraction overlap between rules.
        """
        if self.explainer.scope.value == 'local':
            if not instance:
                instance = self.dataset.X[0]
            rules = self.explainer(instance, n_rules=n_rules)
        else:
            rules = self.explainer(n_rules=n_rules)

        number_of_rules = len(rules)
        # Calculate rule overlap
        overlap_sum = 0
        for rule1, rule2 in combinations(rules, 2):
            overlap_sum += self.calculate_rule_overlap(
                rule1, rule2) / len(self.dataset.X)

        # Calculate fraction overlap
        # Add edge case to avoid division by zero
        denominator = (number_of_rules * (number_of_rules - 1)) or 1
        fraction_overlap = 2 / denominator * overlap_sum
        return fraction_overlap

    def calculate_rule_overlap(self, rule1, rule2, instances=None):
        """
        Calculates the overlap count between two rules.

        Args:
            rule1: The first rule to compare.
            rule2: The second rule to compare.

        Returns:
            The number of instances where both rules apply.

        """
        overlap_count = 0
        if instances is None:
            instances = self.dataset.X_vectorized
        overlap_rule_1 = self.check_rule_applies(
            rule1, instances)
        overlap_rule_2 = self.check_rule_applies(
            rule2, instances)
        overlap_count = [overlap_rule_1[i] and overlap_rule_2[i]
                         for i in range(len(overlap_rule_1))].count(True)
        return overlap_count

    def check_rule_applies(self, rule, instances):
        """
        Checks if a rule applies to a given list of instances.

        Parameters:
        - rule: A dictionary where keys are features (words) and values indicate the presence (1) or absence (0) required in the instance.
        - instance: A list of text string to be checked against the rule.

        Returns:
        - A list of True if the rule applies to the instance, False otherwise.
        """
        instances = np.array(self.dataset.X_vectorized.todense().tolist())
        check_rule = np.ones(len(instances))
        # Apply each condition in the rule
        for feature_name, condition in rule.items():
            # Split the condition into column name, comparison operator, and threshold
            comparison_op, threshold = condition[0], condition[1]

            # Convert threshold to int or float
            threshold = float(threshold)

            # Evaluate the condition and update the mask
            feature_index = np.where(
                self.dataset.feature_names == feature_name)[0][0]
            check_rule *= (instances[:, feature_index] <= threshold) if comparison_op == '<=' else (
                instances[:, feature_index] > threshold)
        return check_rule
