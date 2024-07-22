AVAILABLE_DATASETS = [
    'Imdb',
]

AVAILABLE_EXPLAINERS = [
    'AnchorExplainer',
    'LimeExplainer',
    'ShapExplainer',
    'RandomFeatureImportanceExplainer',
    'RandomRuleExplainer',
    'SurrogateDecisionTreeExplainer',
    'SurrogateLogisticRegressionExplainer',
    'SurrogateMultinomialNBExplainer',
    'BooleanRuleCGExplainer',
]

AVAILABLE_METRICS = [
    'FidelityMetric',
    'NumberRulesMetric',
    'AvgRuleLengthMetric',
    'ChangeSeedMetric',
    'RuleCoverageMetric',
    'RuleOverlapMetric',
    'SensitivityMetric',
    'FaithfulnessCorrelationMetric',
    'ComplexityMetric',
    'AvgCorrectRuleMetric',
]


def available_datasets():
    """
    Returns a list of available datasets.

    Returns:
        list: A list of available datasets.
    """
    return [d for d in AVAILABLE_DATASETS.keys()]


def available_explainers():
    """
    Returns a list of available explainer names.

    Returns:
        list: A list of available explainer names.
    """
    return [e for e in AVAILABLE_EXPLAINERS.keys()]


def available_metrics():
    """
    Returns a list of available metrics.

    Returns:
        list: A list of available metrics.
    """
    return [m for m in AVAILABLE_METRICS.keys()]
