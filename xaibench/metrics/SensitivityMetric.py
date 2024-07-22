from .Metric import Metric
import logging
import pandas as pd
from utilities import ExplanationType, ExplanationScope, MetricCategory, jaccard_similarity, typo_perturb

logger = logging.getLogger(__name__)


class SensitivityMetric(Metric):

    def __init__(self, dataset, explainer, max=True, **kwargs):
        super().__init__(dataset, explainer, **kwargs)
        self.scope = ExplanationScope.ANY
        self.explanation_type = ExplanationType.ANY
        self.metric_category = MetricCategory.CONSISTENCY
        self.max = max
        self.validate_explainer()

    def __call__(self, instance=None, perturb_func=None):
        if perturb_func is None:
            perturb_func = typo_perturb

        if instance is None:
            instance = self.dataset.X[0]

        # Retrieve rules from the explainer
        if self.explainer.scope.value == 'local':
            exp_unchanged = self.explainer(instance)
            perturbed_instances = perturb_func(instance)
            if isinstance(perturbed_instances, list) or isinstance(perturbed_instances, pd.core.Series.Series):
                perturbed_instances = perturbed_instances[0]
            exp_perturbed = self.explainer(perturbed_instances)
        else:
            exp_unchanged = self.explainer()
            perturbed_instances = perturb_func(self.dataset.X)
            self.explainer.dataset.model.fit(
                self.explainer.dataset.vectorizer.fit_transform(perturbed_instances), self.dataset.y)
            exp_perturbed = self.explainer()

        explanation_length = min(len(exp_unchanged), len(exp_perturbed))

        similarities = []
        # Calculate the similarity between the explanations
        if self.explainer.explanation_type == ExplanationType.RULE:
            for i in range(explanation_length):
                similarities.append(
                    jaccard_similarity(exp_unchanged[i], exp_perturbed[i]))
        else:
            for i in range(explanation_length):
                similarities.append(
                    1 - abs(exp_unchanged[i][1] - exp_perturbed[i][1]))

        # Return the average or max similarity
        if self.max:
            return max(similarities)
        return sum(similarities) / len(similarities)
