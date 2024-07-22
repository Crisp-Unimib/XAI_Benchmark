from .Explainer import Explainer
import logging
from utilities import ExplanationType, ExplanationScope

import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

logger = logging.getLogger(__name__)


class SurrogateDecisionTreeExplainer(Explainer):
    """
    An explainer that builds a surrogate model using a decision tree to approximate
    the behavior of a possibly more complex model or system. It uses the fitted model
    to identify and return the most important features.
    """

    def __init__(self, dataset, random_state=42, **kwargs):
        """
        Initializes the SurrogateModelExplainer with dataset features and target labels.

        Parameters:
            dataset: The dataset used for training the model.
            random_state (int, optional): The seed used by the random number generator.
        """
        super().__init__(**kwargs)
        self.dataset = dataset
        self.scope = ExplanationScope.GLOBAL
        self.explanation_type = ExplanationType.RULE
        self.explainer = DecisionTreeClassifier(
            random_state=random_state, max_leaf_nodes=20, max_depth=10)
        self.explainer.fit(self.dataset.X_vectorized, self.dataset.y_predicted)

    def get_rules(self):
        tree_ = self.explainer.tree_
        feature_name = [
            self.dataset.feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        def traverse_nodes(node, current_rule, rules):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                left_child = tree_.children_left[node]
                right_child = tree_.children_right[node]

                # Add condition for the left child
                left_rule = f"{name} <= {threshold:.3f}"
                traverse_nodes(left_child, current_rule + [left_rule], rules)

                # Add condition for the right child
                right_rule = f"{name} > {threshold:.3f}"
                traverse_nodes(right_child, current_rule + [right_rule], rules)
            else:
                # Leaf node
                rule = {
                    'rule': ' & '.join(current_rule),
                    'samples': int(tree_.n_node_samples[node])
                }
                if self.dataset.class_names:
                    class_id = np.argmax(tree_.value[node])
                    class_name = self.dataset.class_names[class_id]
                    rule['class'] = class_name
                else:
                    rule['response'] = tree_.value[node]
                rules.append(rule)

        rules = []
        traverse_nodes(0, [], rules)

        def format_rule(rule):
            rule_features = rule['rule'].split(' & ')
            rule_dict = {}
            for r in rule_features:
                if '<=' in r:
                    rule_dict[r.split(' <= ')[0]] = (
                        '<=', float(r.split(' <= ')[1]))
                else:
                    rule_dict[r.split(' > ')[0]] = (
                        '>', float(r.split(' > ')[1]))
            return rule_dict
        return [format_rule(rule) for rule in rules]

    def __call__(self, n_rules=1000):
        """
        Returns the most important features based on the fitted model.

        Parameters:
            n_rules (int): The number of top features to return.

        Returns:
            numpy.ndarray: Indices of the top n features.
        """
        # Get the indices of the features sorted by importance
        rules = self.get_rules()
        return rules[:n_rules]


class SurrogateLogisticRegressionExplainer(Explainer):
    """
    An explainer that builds a surrogate model using logistic regression to approximate
    the behavior of a possibly more complex model or system. It uses the fitted model
    to identify and return the most important features based on the coefficients.
    """

    def __init__(self, dataset, random_state=42, **kwargs):
        """
        Initializes the SurrogateModelExplainer with a dataset.

        Parameters:
            dataset: The dataset used for training the model, expected to have
                     X_vectorized (features) and y_predicted (target labels).
            random_state (int, optional): The seed used by the random number generator.
        """
        self.dataset = dataset
        self.scope = ExplanationScope.GLOBAL
        self.explanation_type = ExplanationType.FEATURE
        self.explainer = LogisticRegression(
            random_state=random_state, **kwargs)
        self.explainer.fit(self.dataset.X_vectorized, self.dataset.y_predicted)

    def get_important_features(self, n_features=6):
        """
        Identifies the most important features based on the absolute values of
        the model's coefficients.

        Parameters:
            n_features (int): The number of top features to return.

        Returns:
            A list of tuples, each containing the feature name and its coefficient,
            sorted by importance.
        """
        # Get coefficients and feature names
        coefficients = self.explainer.coef_[0]
        feature_names = self.dataset.feature_names

        # Calculate the absolute values of coefficients to determine importance
        importance = np.abs(coefficients)

        # Sort features by importance
        sorted_indices = np.argsort(importance)[::-1]
        important_features = [(feature_names[i], coefficients[i])
                              for i in sorted_indices[:n_features]]
        # Normalise the coefficients between -1 and 1
        max_coeff = max([abs(coeff) for _, coeff in important_features])
        important_features = [(feature, coeff / max_coeff)
                              for feature, coeff in important_features]

        return important_features

    def __call__(self, n_features=6):
        """
        Returns the most important features based on the fitted model's coefficients.

        Parameters:
            n_features (int): The number of top features to return.

        Returns:
            A list of tuples containing the most important features and their coefficients.
        """
        return self.get_important_features(n_features=n_features)


class SurrogateMultinomialNBExplainer(Explainer):
    """
    An explainer that builds a surrogate model using Multinomial Naive Bayes to approximate
    the behavior of a possibly more complex model or system. It aims to use the model's
    feature log probabilities to discuss feature importance.
    """

    def __init__(self, dataset, **kwargs):
        """
        Initializes the MultinomialNBExplainer with a dataset.

        Parameters:
            dataset: The dataset used for training the model, expected to have
                     X_vectorized (features) and y_predicted (target labels).
            random_state (int, optional): The seed used by the random number generator.
        """
        self.dataset = dataset
        self.scope = ExplanationScope.GLOBAL
        self.explanation_type = ExplanationType.FEATURE
        self.explainer = MultinomialNB(**kwargs)
        self.explainer.fit(self.dataset.X_vectorized, self.dataset.y_predicted)

    def get_feature_importance(self, class_index=1, n_features=6):
        """
        Identifies the most important features for a given class, based on the log
        probabilities of features given the class.

        Parameters:
            class_index (int): The index of the class for which to return feature importance.
            n_features (int): The number of top features to return.

        Returns:
            A list of tuples, each containing the feature name and its log probability,
            sorted by importance.
        """
        # Get log probabilities of features given the class
        log_probabilities = self.explainer.feature_log_prob_[class_index]

        # Sort features by log probability
        sorted_indices = np.argsort(log_probabilities)[::-1]
        important_features = [(self.dataset.feature_names[i], log_probabilities[i])
                              for i in sorted_indices[:n_features]]

        return important_features

    @staticmethod
    def log_probs_to_importance(log_probs):
        """
        Convert log probabilities to importance scores.

        Parameters:
        log_probs (numpy.ndarray): Array of log probabilities.

        Returns:
        numpy.ndarray: Array of importance scores.
        """
        probs = np.exp(log_probs)
        # Apply softmax to get normalized importance
        importance = np.exp(probs) / np.sum(np.exp(probs), axis=0)
        return importance

    def __call__(self, class_index=0, n_features=6):
        """
        Returns the most important features for a given class, based on the model's
        feature log probabilities.

        Parameters:
            class_index (int): The index of the class for which to return feature importance.
            n_features (int): The number of top features to return.

        Returns:
            A list of tuples containing the most important features for the class and their log probabilities.
        """
        feature_importance = self.get_feature_importance(
            class_index=class_index, n_features=n_features)

        # Extract just the log probabilities from the list, ignoring the feature names
        log_probs_values = np.array([item[1] for item in feature_importance])
        # Apply the function to our log probabilities
        importance_scores = self.log_probs_to_importance(log_probs_values)

        # Pair each feature with its importance score for easier interpretation
        feature_importance = [
            (feature_importance[i][0], importance_scores[i]) for i in range(len(log_probs_values))]

        return feature_importance
