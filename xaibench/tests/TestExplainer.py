import unittest
from unittest.mock import patch
import numpy as np
from explainers import (
    LimeExplainer,
    ShapExplainer,
    AnchorExplainer,
    RandomFeatureImportanceExplainer,
    RandomRuleExplainer,
    SurrogateDecisionTreeExplainer,
    SurrogateLogisticRegressionExplainer,
    SurrogateMultinomialNBExplainer,
)
from data import Imdb
from utilities import ExplanationType, ExplanationScope

test_shap_values = [
    np.array(
        [np.array([-0.03, 0.02, -0.01, -0.02, -0.01, 0.03, -0.0, 0.02, 0.01, -0.02])]),
    np.array(
        [np.array([0.03, -0.02, 0.01, 0.02, 0.01, -0.03, 0.0, -0.02, -0.01, 0.02])])
]


class BaseFeatureExplainerTestCase(unittest.TestCase):
    """Superclass for shared explainer test functionality."""

    def setUp(self):
        self.dataset = Imdb(num_features=10)
        self.explainer = None
        # Initialize explainer in subclass setup

    def test_explanation_scope(self):
        """
        Test the ExplanationScope attribute of the explainer.
        """
        self.assertIsInstance(
            self.explainer.scope, ExplanationScope)
        self.assertIn(self.explainer.scope.value,
                      ['local', 'global', 'any'])

    def test_explanation_type(self):
        """
        Test the explanation_type attribute of the explainer.
        """
        self.assertIsInstance(self.explainer.explanation_type, ExplanationType)
        self.assertEqual(self.explainer.explanation_type.value,
                         'feature_importance')

    def test_call_output_length(self):
        """
        Test the length of the output from the explainer's call method.
        """
        if self.explainer.scope.value == 'local':
            explanation = self.explainer(self.dataset.X[0], n_features=5)
        else:
            explanation = self.explainer(n_features=5)
        self.assertEqual(len(explanation), 5)

    def test_call_output_content_types(self):
        """
        Test the content types of the output from the explainer's call method.
        """
        if self.explainer.scope.value == 'local':
            explanation = self.explainer(self.dataset.X[0])
        else:
            explanation = self.explainer()
        for feature_name, value in explanation:
            self.assertIsInstance(feature_name, str)
            self.assertIsInstance(value, float)

    def test_call_output_value_range(self):
        """
        Test the output value range of the explainer's call method.
        The explanation values should be within the range of -1.0 to 1.0.
        """
        if self.explainer.scope.value == 'local':
            explanation = self.explainer(self.dataset.X[0])
        else:
            explanation = self.explainer()
        for _, value in explanation:
            self.assertTrue(-1.0 <= value <= 1.0)

    def test_call_with_excess_n_features(self):
        """Test call with more features than dataset contains."""
        if self.explainer.scope.value == 'local':
            explanation = self.explainer(self.dataset.X[0], n_features=20)
        else:
            explanation = self.explainer(n_features=20)
        self.assertTrue(len(explanation) <= 20)


class TestLimeExplainer(BaseFeatureExplainerTestCase):
    def setUp(self):
        super().setUp()
        self.explainer = LimeExplainer(self.dataset)

    def test_init(self):
        """
        Test case for initializing the explainer.

        This test checks if the dataset attribute is set correctly and if the explainer object is not None.
        """
        self.assertEqual(self.explainer.dataset, self.dataset)
        self.assertIsNotNone(self.explainer.explainer)

    @patch('lime.lime_text.LimeTextExplainer.explain_instance')
    def test_call(self, mock_explain_instance):
        """
        Test the call method of the Explainer class.

        Args:
            mock_explain_instance: Mock object for explain_instance method.

        Returns:
            None
        """
        mock_explain_instance.return_value.as_list.return_value = [
            ('feature1', 0.1), ('feature2', 0.2)]
        instance = 'test instance'
        result = self.explainer(instance)
        mock_explain_instance.assert_called_once_with(
            instance, self.dataset.pipe.predict_proba, num_features=6)
        self.assertEqual(result, [('feature1', 0.1), ('feature2', 0.2)])


class TestShapExplainer(BaseFeatureExplainerTestCase):
    def setUp(self):
        super().setUp()
        self.explainer = ShapExplainer(self.dataset)

    def test_init(self):
        """
        Test case for initializing the explainer.

        This method checks if the dataset attribute is set correctly and if the explainer object is not None.
        """
        self.assertEqual(self.explainer.dataset, self.dataset)
        self.assertIsNotNone(self.explainer.explainer)

    @patch('shap.KernelExplainer.shap_values')
    def test_call(self, mock_shap_values):
        """
        Test the call method of the Explainer class.

        Args:
            mock_shap_values: A mock object for the shap_values function.

        Returns:
            None
        """

        mock_shap_values.return_value = test_shap_values

        instance = 'test instance'
        n_features = 6
        result = self.explainer(instance, n_features)
        self.assertEqual(len(result), n_features)

    @patch('shap.KernelExplainer.shap_values')
    def test_call_with_different_n_features(self, mock_shap_values):
        """
        Test case for the `test_call_with_different_n_features` method.

        This test verifies that the `explainer` method returns the expected result
        when called with a different number of features.

        Args:
            mock_shap_values: A mock object for the `shap_values` function.

        Returns:
            None
        """

        mock_shap_values.return_value = test_shap_values
        instance = 'test instance'
        n_features = 3
        result = self.explainer(instance, n_features)
        self.assertEqual(len(result), n_features)


class TestRandomFeatureImportanceExplainer(BaseFeatureExplainerTestCase):
    def setUp(self):
        super().setUp()
        self.explainer = RandomFeatureImportanceExplainer(self.dataset)

    def test_init(self):
        """
        Test case for initializing the explainer.
        """
        self.assertEqual(self.explainer.dataset, self.dataset)


class TestSurrogateLogisticRegressionExplainer(BaseFeatureExplainerTestCase):
    def setUp(self):
        super().setUp()
        self.explainer = SurrogateLogisticRegressionExplainer(self.dataset)

    def test_init(self):
        """
        Test case for initializing the explainer.
        """
        self.assertEqual(self.explainer.dataset, self.dataset)
        self.assertIsNotNone(self.explainer.explainer)


class TestSurrogateMultinomialNBExplainer(BaseFeatureExplainerTestCase):
    def setUp(self):
        super().setUp()
        self.explainer = SurrogateMultinomialNBExplainer(self.dataset)

    def test_init(self):
        """
        Test case for initializing the explainer.
        """
        self.assertEqual(self.explainer.dataset, self.dataset)
        self.assertIsNotNone(self.explainer.explainer)


class BaseRuleExplainerTestCase(unittest.TestCase):
    """Superclass for shared explainer test functionality."""

    def setUp(self):
        self.dataset = Imdb(num_features=10)
        self.explainer = None
        # Initialize explainer in subclass setup

    def test_explanation_type(self):
        """
        Test the explanation_type attribute of the explainer.
        """
        self.assertIsInstance(self.explainer.explanation_type, ExplanationType)
        self.assertEqual(self.explainer.explanation_type.value, 'rule_based')

    def test_explanation_scope(self):
        """
        Test the ExplanationScope attribute of the explainer.
        """
        self.assertIsInstance(
            self.explainer.scope, ExplanationScope)
        self.assertIn(self.explainer.scope.value,
                      ['global', 'local', 'any'])

    def test_call_output_length(self):
        """
        Test the length of the output from the explainer's call method.
        """
        if self.explainer.scope.value == 'local':
            explanation = self.explainer(self.dataset.X[0], n_rules=5)
        else:
            explanation = self.explainer(n_rules=5)
        self.assertEqual(len(explanation), 5)

    def test_call_output_content_types(self):
        """
        Test the content types of the output from the explainer's call method.
        """
        if self.explainer.scope.value == 'local':
            explanation = self.explainer(self.dataset.X[0])
        else:
            explanation = self.explainer()
        for rule in explanation:
            self.assertIsInstance(rule, dict)
            for feature, value in rule.items():
                self.assertIsInstance(feature, str)
                self.assertIsInstance(value, int)
                self.assertIn(value, [0, 1])


class TestRandomRuleExplainer(BaseRuleExplainerTestCase):
    def setUp(self):
        super().setUp()
        self.explainer = RandomRuleExplainer(self.dataset)

    def test_init(self):
        """
        Test case for initializing the explainer.
        """
        self.assertEqual(self.explainer.dataset, self.dataset)


class TestAnchorExplainer(BaseRuleExplainerTestCase):
    def setUp(self):
        super().setUp()
        self.explainer = AnchorExplainer(self.dataset)

    def test_init(self):
        """
        Test case for initializing the explainer.
        """
        self.assertEqual(self.explainer.dataset, self.dataset)


class TestSurrogateDecisionTreeExplainer(BaseRuleExplainerTestCase):
    def setUp(self):
        super().setUp()
        self.explainer = SurrogateDecisionTreeExplainer(self.dataset)

    def test_init(self):
        """
        Test case for initializing the explainer.
        """
        self.assertEqual(self.explainer.dataset, self.dataset)
        self.assertIsNotNone(self.explainer.explainer)


if __name__ == '__main__':
    unittest.main()
