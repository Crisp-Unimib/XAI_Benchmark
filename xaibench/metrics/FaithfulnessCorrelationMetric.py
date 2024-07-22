from .Metric import Metric
import logging
import pandas as pd
from itertools import chain
from scipy import stats
from utilities import ExplanationType, ExplanationScope, MetricCategory, baseline_perturb

logger = logging.getLogger(__name__)


class FaithfulnessCorrelationMetric(Metric):
    """
    Calculates the faithfulness correlation metric for a given set of instances.

    Inherits from Metric class.

    Attributes:
        dataset: The dataset to evaluate the metric on.
        explainer: The explainer used to generate explanations.
        n_runs (optional): The number of runs to perform for the metric calculation. Default is 2.
        subset_size (optional): The size of the subset of instances to evaluate the metric on. Default is 3.
    """

    def __init__(self, dataset, explainer, n_runs=2, subset_size=3, **kwargs):
        """
        Initializes a FaithfulnessCorrelationMetric object.

        Args:
            dataset: The dataset to evaluate the metric on.
            explainer: The explainer used to generate explanations.
            n_runs (optional): The number of runs to perform for the metric calculation. Default is 2.
            subset_size (optional): The size of the subset of instances to evaluate the metric on. Default is 3.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        super().__init__(dataset, explainer, **kwargs)
        self.scope = ExplanationScope.LOCAL
        self.explanation_type = ExplanationType.FEATURE
        self.metric_category = MetricCategory.COMPLETENESS
        self.n_runs = n_runs
        self.subset_size = subset_size
        self.validate_explainer()

    def __call__(self, instances=None, perturb_func=None):
        """
        Calculates the faithfulness correlation metric for a given set of instances.

        Parameters:
        instances (list): List of instances to evaluate the metric on.
        perturb_func (function, optional): Function to perturb the instances. Defaults to baseline_perturb.

        Returns:
        float: The faithfulness correlation metric value.
        """
        if perturb_func is None:
            perturb_func = baseline_perturb
        if instances is None:
            instances = self.dataset.X[:5]

        explanations = [self.explainer(instance, n_features=20)
                        for instance in instances]
        y_pred = self.dataset.model.predict_proba(
            self.dataset.vectorizer.transform(instances))

        pred_deltas = []
        att_sums = []
        for _ in range(self.n_runs):
            masked_instances = perturb_func(instances)
            y_pred_perturb = self.explainer.dataset.model.predict_proba(
                self.explainer.dataset.vectorizer.transform(masked_instances))

            pred_deltas.append(
                [float(y[1] - y_p[1]) for y, y_p in zip(y_pred, y_pred_perturb)])

            perturbed_words = [list(set(i.split()) - set(p_i.split()))
                               for i, p_i in zip(instances, masked_instances)]

            att_sums.append([sum([dict(explanations[i]).get(
                word, 0) for word in words]) for i, words in enumerate(perturbed_words)])

        pred_deltas = list(chain(*pred_deltas))
        att_sums = list(chain(*att_sums))
        result = stats.pearsonr(pred_deltas, att_sums)
        return result[0]
