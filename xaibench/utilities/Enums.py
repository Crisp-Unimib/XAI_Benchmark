from enum import Enum


class ExplanationType(Enum):
    """
    Enum class representing different types of output in the XAI benchmark.
    """
    FEATURE = 'feature_importance'
    RULE = 'rule_based'
    ANY = 'any'


class ExplanationScope(Enum):
    """
    Enum class representing different explanation scopes for the XAI benchmark.
    """
    GLOBAL = 'global'
    LOCAL = 'local'
    ANY = 'any'


class MetricCategory(Enum):
    """
    Enum representing different categories of metrics.
    """

    SOUNDNESS = 'soundness'
    COMPLETENESS = 'completeness'
    CONTEXTFULNESS = 'contextfulness'
    NOVELTY = 'novelty'
    PARSIMONY = 'parsimony'
    CONSISTENCY = 'consistency'
    STABILITY = 'stability'
    ANY = 'any'
