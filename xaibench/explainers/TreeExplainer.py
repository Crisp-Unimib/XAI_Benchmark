from .Explainer import Explainer
import logging
import json
from utilities import ExplanationType, ExplanationScope
from chefboost import Chefboost as chef

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TreeWrapper():
    def __init__(self, config, feature_names):
        self.config = config
        self.feature_names = feature_names
        self.model = None

    def fit(self, x, y, **kwargs):
        data = pd.DataFrame(x.todense(),
                            columns=self.feature_names)
        data['target'] = y
        self.model = chef.fit(data, config=self.config, target_label='target')
        return self.model

    def predict(self, instances):
        module_name = 'outputs/rules/rules'
        tree = chef.restoreTree(module_name)
        return [tree.findDecision(instance.toarray()[0]) for instance in instances]

    def predict_proba(self, instances):
        module_name = 'outputs/rules/rules'
        tree = chef.restoreTree(module_name)
        predictions = [tree.findDecision(
            instance.toarray()[0]) for instance in instances]
        probas = list(
            np.zeros((instances.shape[0], 2)))
        for i, prediction in enumerate(predictions):
            probas[i][0] = 1 - prediction
            probas[i][1] = prediction
        return np.array(probas)


def get_rules(nodes):
    def traverse_nodes(current_node, current_rule, rules):
        children = [node for node in nodes if node['parents'] == current_node]

        for child in children:
            name = child['feature_name']
            if name == '':
                continue

            if child['return_statement'] == 1:
                # Leaf node
                if child['rule'] == 'return 1':
                    rules.append(' & '.join(current_rule))
                return 'Leaf'

            if '<=' in child['rule']:
                threshold = child['rule'].split('<=')[1].strip(':')
                condition = '<='
            else:
                threshold = child['rule'].split('>')[1].strip(':')
                condition = '>'
            rule = f"{name} {condition} {threshold}"

            traverse_nodes(child['leaf_id'], current_rule + [rule], rules)

    rules = []

    def format_rule(rule):
        rule_features = rule.split(' & ')
        rule_dict = {}
        for r in rule_features:
            if '<=' in r:
                rule_dict[r.split(' <= ')[0]] = ('<=', r.split(' <= ')[1])
            else:
                rule_dict[r.split(' > ')[0]] = ('>', r.split(' > ')[1])
        return rule_dict
    traverse_nodes('root', [], rules)
    return [format_rule(rule) for rule in rules]


class C45Explainer(Explainer):

    def __init__(self, dataset, random_state=42, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.scope = ExplanationScope.GLOBAL
        self.explanation_type = ExplanationType.RULE
        config = {'algorithm': 'C4.5',
                  'max_depth': 10, 'enableParallelism': True}
        self.explainer = TreeWrapper(config, self.dataset.feature_names)
        self.explainer.fit(self.dataset.X_vectorized,
                           self.dataset.y_predicted, silent=True)

    def __call__(self, n_rules=20):
        """
        Returns the most important features based on the fitted model.

        Parameters:
            n_rules (int): The number of top features to return.

        Returns:
            numpy.ndarray: Indices of the top n features.
        """
        # Get the indices of the features sorted by importance
        # rules = chef.feature_importance('outputs/rules/rules.py')
        # feature_importances = []
        # for feature, rule in zip(rules['feature'], rules['importance']):
        #     feature_importances.append((feature, rule))
        # return feature_importances[:n_rules]

        # Load the JSON data from the file
        with open('outputs/rules/rules.json', 'r') as file:
            data = json.load(file)
        rules_list = get_rules(data)
        return rules_list[:n_rules]


class ID3Explainer(Explainer):

    def __init__(self, dataset, random_state=42, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.scope = ExplanationScope.GLOBAL
        self.explanation_type = ExplanationType.RULE
        config = {'algorithm': 'ID3', 'max_depth': 10,
                  'enableParallelism': True}
        self.explainer = TreeWrapper(config, self.dataset.feature_names)
        self.explainer.fit(self.dataset.X_vectorized,
                           self.dataset.y_predicted, silent=True)

    def __call__(self, n_rules=20):
        """
        Returns the most important features based on the fitted model.

        Parameters:
            n_features (int): The number of top features to return.

        Returns:
            numpy.ndarray: Indices of the top n features.
        """
        with open('outputs/rules/rules.json', 'r') as file:
            data = json.load(file)
        rules_list = get_rules(data)
        return rules_list[:n_rules]


class CHAIDExplainer(Explainer):

    def __init__(self, dataset, random_state=42, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.scope = ExplanationScope.GLOBAL
        self.explanation_type = ExplanationType.RULE
        config = {'algorithm': 'CHAID',
                  'max_depth': 10, 'enableParallelism': True}
        self.explainer = TreeWrapper(config, self.dataset.feature_names)
        self.explainer.fit(self.dataset.X_vectorized,
                           self.dataset.y_predicted, silent=True)

    def __call__(self, n_rules=20):
        """
        Returns the most important features based on the fitted model.

        Parameters:
            n_rules (int): The number of top features to return.

        Returns:
            numpy.ndarray: Indices of the top n features.
        """
        with open('outputs/rules/rules.json', 'r') as file:
            data = json.load(file)
        rules_list = get_rules(data)
        return rules_list[:n_rules]
