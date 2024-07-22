from .Metric import Metric
import logging
import numpy as np
from utilities import ExplanationType, ExplanationScope, MetricCategory

logger = logging.getLogger(__name__)


class AvgCorrectRuleMetric(Metric):

    def __init__(self, dataset, explainer, **kwargs):
        super().__init__(dataset, explainer, **kwargs)
        self.scope = ExplanationScope.LOCAL
        self.explanation_type = ExplanationType.RULE
        self.metric_category = MetricCategory.COMPLETENESS
        self.validate_explainer()

    def check_rule_applies(self, rule, instance):
        """
        Checks if a rule applies to a given instance.

        Parameters:
        - rule: A dictionary where keys are features (words) and values indicate the presence (1) or absence (0) required in the instance.
        - instance: A text string to be checked against the rule.

        Returns:
        - True if the rule applies to the instance, False otherwise.
        """
        # Initialize a boolean
        applies = True
        instance = instance.todense().tolist()[0]

        # Apply each condition in the rule
        for feature_name, condition in rule.items():
            # Split the condition into column name, comparison operator, and threshold
            comparison_op, threshold = condition[0], condition[1]

            # Convert threshold to int or float
            threshold = float(threshold)

            # Evaluate the condition and update the mask
            feature_index = np.where(
                self.dataset.feature_names == feature_name)[0][0]
            applies &= (instance[feature_index] <= threshold) if comparison_op == '<=' else (
                instance[feature_index] > threshold)
        return applies

    def __call__(self, instance=None):
        if instance is None:
            instance = self.dataset.X_vectorized[0:10]

        # Retrieve rules from the explainer
        rules = [self.explainer(i) for i in instance]

        # Check if the rule applies to every positive instance in the dataset
        correct_rule_counts = []
        for rule in rules:
            running_total = 0
            running_den = 0
            if isinstance(rule, list):
                rule = rule[0]
            for instance_x, instance_y in zip(self.dataset.X_vectorized, self.dataset.y):
                if instance_y == 0:
                    continue
                rule_applies = self.check_rule_applies(rule, instance_x)
                if rule_applies == instance_y:
                    running_total += 1
                running_den += 1
            correct_rule_counts.append(running_total / max(running_den, 1))
        return np.mean(correct_rule_counts)
