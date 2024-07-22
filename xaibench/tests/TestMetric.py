import unittest
from unittest.mock import patch
import numpy as np
from itertools import combinations
from explainers import (
    RandomFeatureImportanceExplainer,
    RandomRuleExplainer,
    LimeExplainer,
    SurrogateDecisionTreeExplainer,
)
from data import Imdb
from metrics import (
    FidelityMetric,
    NumberRulesMetric,
    AvgRuleLengthMetric,
    ChangeSeedMetric,
    RuleCoverageMetric,
    RuleOverlapMetric,
    SensitivityMetric,
    FaithfulnessCorrelationMetric,
    ComplexityMetric,
    AvgCorrectRuleMetric,
)
from utilities import ExplanationType, ExplanationScope, typo_perturb


class BaseMetricTestCase(unittest.TestCase):
    """Superclass for shared metric test functionality."""

    def setUp(self):
        self.dataset = Imdb(num_features=10)
        self.explainer = None
        # Metric classes should be initialized with a dataset and explainer
        # Create subclasses that inherit from the Metric class for each metric
        self.metric = None

    def test_metric_scope(self):
        """
        Test the ExplanationScope attribute of the metric.
        """
        self.assertIsInstance(
            self.metric.scope, ExplanationScope)
        self.assertIn(self.metric.scope.value,
                      ['local', 'global', 'any'])

    def test_explanation_type_accepted(self):
        """
        Test the explanation_type attribute of the metric.
        """
        self.assertIsInstance(self.metric.explanation_type, ExplanationType)
        self.assertIn(self.metric.explanation_type.value,
                      ['feature_importance', 'rule_based', 'any'])

    def test_validate_explainer(self):
        """
        Test the validate_explainer method.
        """
        try:
            self.metric.validate_explainer()
            validation_passed = True
        except ValueError as e:
            validation_passed = False

        self.assertTrue(validation_passed,
                        'validate_explainer should not raise an error')


class TestFidelityMetric(BaseMetricTestCase):
    def setUp(self):
        super().setUp()
        self.explainer = RandomFeatureImportanceExplainer(self.dataset)
        self.metric = FidelityMetric(self.dataset, self.explainer)

    def test_init(self):
        """
        Test case for initializing the metric.
        """
        self.assertIsInstance(self.metric, FidelityMetric)


class TestNumberRulesMetric(BaseMetricTestCase):
    def setUp(self):
        super().setUp()
        self.explainer = RandomRuleExplainer(self.dataset)
        self.metric = NumberRulesMetric(self.dataset, self.explainer)

    def test_init(self):
        """
        Test case for initializing the metric.
        """
        self.assertIsInstance(self.metric, NumberRulesMetric)


class TestAvgRuleLengthMetric(BaseMetricTestCase):
    def setUp(self):
        super().setUp()
        self.explainer = RandomRuleExplainer(self.dataset)
        self.metric = AvgRuleLengthMetric(self.dataset, self.explainer)

    def test_init(self):
        """
        Test case for initializing the metric.
        """
        self.assertIsInstance(self.metric, AvgRuleLengthMetric)


class TestChangeSeedMetric(BaseMetricTestCase):
    def setUp(self):
        super().setUp()
        self.explainer = LimeExplainer(self.dataset)
        self.metric = ChangeSeedMetric(self.dataset, self.explainer)

    def test_init(self):
        """
        Test case for initializing the metric.
        """
        self.assertIsInstance(self.metric, ChangeSeedMetric)


class TestRuleCoverageMetric(BaseMetricTestCase):
    def setUp(self):
        super().setUp()
        self.explainer = RandomRuleExplainer(self.dataset)
        self.metric = RuleCoverageMetric(self.dataset, self.explainer)

    def test_init(self):
        """
        Test case for initializing the metric.
        """
        self.assertIsInstance(self.metric, RuleCoverageMetric)

    def test_rule_coverage(self):
        """
        Test the rule coverage for the Metric class.

        This test checks if the rule coverage is correctly calculated by the Metric class.
        It creates a rule and a list of instances, and then checks if the rule applies to each instance.
        The expected results are [True, False, False].

        """
        rule = {'example': ('>=', 1), 'missing': ('<', 0)}
        instances = [
            'this is an example sentence',
            'this sentence lacks both keywords',
            'this is an example with the missing word'
        ]
        results = [self.metric.check_rule_applies(
            rule, instance) for instance in instances]
        self.assertEqual(results, [True, False, False])


class TestRuleOverlapMetric(BaseMetricTestCase):
    def setUp(self):
        super().setUp()
        self.explainer = RandomRuleExplainer(self.dataset)
        self.metric = RuleOverlapMetric(self.dataset, self.explainer)

    def test_init(self):
        """
        Test case for initializing the metric.
        """
        self.assertIsInstance(self.metric, RuleOverlapMetric)

    def test_rule_overlap(self):
        """
        Test case for calculating rule overlap.

        This test case checks the correctness of the `calculate_rule_overlap` method
        by providing a set of rules and dataset instances. It calculates the overlap
        for each combination of rules and asserts that the calculated overlap counts
        match the expected values.

        """
        rules = [
            {'example': 1, 'test': 0},
            {'another': 1, 'fail': 1}
        ]
        dataset_instances = [
            'this is an example sentence. do you think it will fail ? maybe another time',
            'for example this should fail in the second rule',
            'this is another example that should not fail'
        ]

        # Calculate overlap for each combination of rules
        overlap_counts = [self.metric.calculate_rule_overlap(
            rules[0], rules[1], [instance]) for instance in dataset_instances]
        self.assertEqual(overlap_counts, [1, 0, 1])


class TestSensitivityMetric(BaseMetricTestCase):
    def setUp(self):
        super().setUp()
        self.explainer = RandomRuleExplainer(self.dataset)
        self.metric = SensitivityMetric(self.dataset, self.explainer)

    def test_init(self):
        """
        Test case for initializing the metric.
        """
        self.assertIsInstance(self.metric, SensitivityMetric)


class TestFaithfulnessCorrelationMetric(BaseMetricTestCase):
    def setUp(self):
        super().setUp()
        self.explainer = RandomFeatureImportanceExplainer(self.dataset)
        self.metric = FaithfulnessCorrelationMetric(
            self.dataset, self.explainer)

    def test_init(self):
        """
        Test case for initializing the metric.
        """
        self.assertIsInstance(self.metric, FaithfulnessCorrelationMetric)


class TestComplexityMetric(BaseMetricTestCase):
    def setUp(self):
        super().setUp()
        self.explainer = RandomFeatureImportanceExplainer(self.dataset)
        self.metric = ComplexityMetric(
            self.dataset, self.explainer)

    def test_init(self):
        """
        Test case for initializing the metric.
        """
        self.assertIsInstance(self.metric, ComplexityMetric)


class TestAvgCorrectRuleMetric(BaseMetricTestCase):
    def setUp(self):
        super().setUp()
        self.explainer = RandomRuleExplainer(self.dataset)
        self.metric = AvgCorrectRuleMetric(
            self.dataset, self.explainer)

    def test_init(self):
        """
        Test case for initializing the metric.
        """
        self.assertIsInstance(self.metric, AvgCorrectRuleMetric)


if __name__ == '__main__':
    unittest.main()
