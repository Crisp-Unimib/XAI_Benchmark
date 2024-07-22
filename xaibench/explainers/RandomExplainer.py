from .Explainer import Explainer
import logging
from utilities import ExplanationType, ExplanationScope
import numpy as np

from random import random, randint, choice

logger = logging.getLogger(__name__)


class RandomWrapper():
    def __init__(self, class_names):
        self.class_names = class_names

    def fit(self, **kwargs):
        pass

    def predict(self, instances):
        return [choice(self.class_names) for _ in range(len(instances))]

    def predict_proba(self, instances):
        return [[random() for _ in self.class_names] for _ in range(instances.shape[0])]


class RandomFeatureImportanceExplainer(Explainer):
    """
    A class that extends the Explainer class to provide random explanations for a dataset.

    This class generates a list of tuples, each containing a randomly selected feature name
    from the dataset and a random float value. The number of tuples generated can be specified.
    """

    def __init__(self, dataset, **kwargs):
        """
        Initializes the RandomFeatureImportanceExplainer instance with a dataset.

        Parameters:
            dataset: The dataset to explain. It is expected to have an attribute 'feature_names'
                     that is indexable and provides the names of the features.
            **kwargs: Additional keyword arguments passed to the superclass's initializer.
        """
        super().__init__(**kwargs)
        self.scope = ExplanationScope.ANY
        self.explanation_type = ExplanationType.FEATURE
        self.dataset = dataset
        self.explainer = RandomWrapper(np.array(self.dataset.class_names))

    def __call__(self, instance=None, n_features=6):
        """
        Generates random explanations for the dataset.

        This method creates and returns a list of tuples. Each tuple contains a randomly
        selected feature name from the dataset and a random float value. The number of tuples
        returned is determined by the `n_features` parameter.

        Parameters:
            n_features (int): The number of features (and thus tuples) to generate. Defaults to 6.

        Returns:
            list of tuples: A list where each tuple contains a feature name and a random float.
        """
        features = []
        if n_features is None:
            # Set the max to the number of features in the dataset
            n_features = len(self.dataset.feature_names)
        for _ in range(n_features):
            features.append(
                (choice(self.dataset.feature_names), 2 * random() - 1))
        return features


class RandomRuleExplainer(Explainer):
    """
    A class that extends the Explainer class to provide random rules for a dataset.

    This class generates a list of strings, each representing a random rule based on a randomly
    selected feature name from the dataset and a random condition. The number of rules generated
    can be specified.
    """

    def __init__(self, dataset, **kwargs):
        """
        Initializes the RandomRuleExplainer instance with a dataset.

        Parameters:
            dataset: The dataset to explain. It is expected to have an attribute 'feature_names'
                     that is indexable and provides the names of the features.
            **kwargs: Additional keyword arguments passed to the superclass's initializer.
        """
        super().__init__(**kwargs)
        self.dataset = dataset
        self.scope = ExplanationScope.ANY
        self.explanation_type = ExplanationType.RULE
        self.explainer = RandomWrapper(np.array(self.dataset.class_names))

    def __call__(self, instance=None, n_rules=6):
        """
        Generates random rules for the dataset.

        This method creates and returns a list of dicts. Each dict represents a rule based on
        a randomly selected feature name from the dataset and a random condition. The number of
        rules returned is determined by the `n_rules` parameter.

        Parameters:
            n_rules (int): The number of rules to generate. Defaults to 6.

        Returns:
            list of dict: A list where each dict represents a random rule.
        """
        rules = []
        n_rules = min(n_rules, 100)
        for _ in range(n_rules):
            rule = {}
            for _ in range(randint(1, 3)):
                feature = choice(self.dataset.feature_names)
                condition = randint(0, 1)
                rule[feature] = condition
            rules.append(rule)
        return rules
