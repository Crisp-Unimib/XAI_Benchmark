from .Metric import Metric
import logging
from time import time
import numpy as np
from utilities import ExplanationType, ExplanationScope, MetricCategory
logger = logging.getLogger(__name__)


class ExecutionTimeMetric(Metric):

    def __init__(self, dataset, explainer, **kwargs):
        """
        Initializes a ExecutionTimeMetric object.

        Args:
            dataset (Dataset): The dataset used for evaluation.
            explainer (Explainer): The explainer used for generating explanations.
            **kwargs: Additional keyword arguments.

        """
        super().__init__(dataset, explainer, **kwargs)
        self.scope = ExplanationScope.ANY
        self.explanation_type = ExplanationType.ANY
        self.metric_category = MetricCategory.PARSIMONY
        self.validate_explainer()

    def __call__(self, instance=None, n_runs=5):
        if instance is None:
            instance = self.dataset.X[0:n_runs]

        times = []
        for i in range(n_runs):
            start_time = time()
            if self.explainer.scope.value in ['local', 'any']:
                self.explainer(instance[i])
            else:
                self.explainer.explainer.fit(
                    self.dataset.X_vectorized, self.dataset.y_predicted)
                self.explainer()
            final_time = time() - start_time
            times.append(final_time)
            if final_time > 15:
                break

        return np.average(times)
