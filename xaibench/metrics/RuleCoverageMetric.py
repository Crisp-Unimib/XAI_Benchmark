from .Metric import Metric
import logging
import numpy as np
from utilities import ExplanationType, ExplanationScope, MetricCategory

logger = logging.getLogger(__name__)


class RuleCoverageMetric(Metric):

    def __init__(self, dataset, explainer, **kwargs):
        """
        Initializes a RuleCoverageMetric object.

        Args:
            dataset: The dataset to evaluate the explanations on.
            explainer: The explainer used to generate the explanations.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(dataset, explainer, **kwargs)
        self.scope = ExplanationScope.ANY
        self.explanation_type = ExplanationType.RULE
        self.metric_category = MetricCategory.CONTEXTFULNESS
        self.validate_explainer()

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

    def __call__(self, instance=None):
        if instance is None:
            instance = self.dataset.X[0:10]

        def find_coverage(rules):
            # Assuming dataset instances can be iterated and compared against rules
            # And assuming 'apply_rule' is a method to check if a rule applies to an instance
            covered_instances = np.zeros(self.dataset.X_vectorized.shape[0])
            for rule in rules:
                covered_instances += self.check_rule_applies(
                    rule, self.dataset.X_vectorized)

            coverage = sum([min(1, x)
                           for x in covered_instances]) / len(self.dataset.X)
            return coverage

        # Retrieve rules from the explainer
        if self.explainer.scope.value == 'local':
            rules = [self.explainer(i) for i in instance]
            coverage = np.average([find_coverage(r) for r in rules])
        else:
            rules = self.explainer()
            coverage = find_coverage(rules)

        return coverage
