import unittest
from tests import (
    TestImdb,
    TestRandomFeatureImportanceExplainer,
    TestLimeExplainer,
    TestShapExplainer,
    TestAnchorExplainer,
    TestSurrogateLogisticRegressionExplainer,
    TestSurrogateMultinomialNBExplainer,
    TestRandomRuleExplainer,
    TestSurrogateDecisionTreeExplainer,
    TestFidelityMetric,
    TestNumberRulesMetric,
    TestAvgRuleLengthMetric,
    TestChangeSeedMetric,
    TestRuleCoverageMetric,
    TestRuleOverlapMetric,
    TestSensitivityMetric,
    TestFaithfulnessCorrelationMetric,
    TestComplexityMetric,
    TestAvgCorrectRuleMetric,
)


def add_test_datasets(suite):
    suite.addTest(unittest.makeSuite(TestImdb))


def add_test_explainers(suite):
    # Feature based
    suite.addTest(unittest.makeSuite(TestRandomFeatureImportanceExplainer))
    suite.addTest(unittest.makeSuite(TestLimeExplainer))
    suite.addTest(unittest.makeSuite(TestShapExplainer))
    suite.addTest(unittest.makeSuite(TestSurrogateLogisticRegressionExplainer))
    suite.addTest(unittest.makeSuite(TestSurrogateMultinomialNBExplainer))

    # Rule based
    suite.addTest(unittest.makeSuite(TestRandomRuleExplainer))
    suite.addTest(unittest.makeSuite(TestAnchorExplainer))
    suite.addTest(unittest.makeSuite(TestSurrogateDecisionTreeExplainer))


def add_test_metrics(suite):
    suite.addTest(unittest.makeSuite(TestAvgCorrectRuleMetric))
    suite.addTest(unittest.makeSuite(TestAvgRuleLengthMetric))
    suite.addTest(unittest.makeSuite(TestChangeSeedMetric))
    suite.addTest(unittest.makeSuite(TestComplexityMetric))
    suite.addTest(unittest.makeSuite(TestFaithfulnessCorrelationMetric))
    suite.addTest(unittest.makeSuite(TestFidelityMetric))
    suite.addTest(unittest.makeSuite(TestNumberRulesMetric))
    suite.addTest(unittest.makeSuite(TestRuleCoverageMetric))
    suite.addTest(unittest.makeSuite(TestRuleOverlapMetric))
    suite.addTest(unittest.makeSuite(TestSensitivityMetric))


if __name__ == '__main__':
    # Create a test suite
    suite = unittest.TestSuite()

    # Add tests to the test suite
    add_test_datasets(suite)
    add_test_explainers(suite)
    add_test_metrics(suite)

    # Run the test suite
    runner = unittest.TextTestRunner(verbosity=2, warnings='ignore')
    runner.run(suite)
