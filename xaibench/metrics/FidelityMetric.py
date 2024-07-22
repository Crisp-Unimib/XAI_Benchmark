from .Metric import Metric
import logging
from sklearn.metrics import f1_score
import numpy as np
from utilities import ExplanationType, ExplanationScope, MetricCategory

logger = logging.getLogger(__name__)


class FidelityMetric(Metric):
    """
    FidelityMetric class calculates the fidelity metric between the original model predictions and the explainer predictions.

    Args:
        dataset: The dataset used for evaluation.
        explainer: The explainer used for generating explanations.
        **kwargs: Additional keyword arguments.

    Attributes:
        scope: The scope of the explanation (GLOBAL, LOCAL, etc.).
        explanation_type: The type of explanation (ANY, FEATURE_IMPORTANCE, etc.).
        metric_category: The category of the metric (SOUNDNESS, COMPLETENESS, etc.).

    Methods:
        __call__: Calculates the fidelity metric between the original model predictions and the explainer predictions.
    """

    def __init__(self, dataset, explainer, **kwargs):
        """
        Initializes a FidelityMetric object.

        Args:
            dataset: The dataset used for evaluation.
            explainer: The explainer used for generating explanations.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(dataset, explainer, **kwargs)
        self.scope = ExplanationScope.GLOBAL
        self.explanation_type = ExplanationType.ANY
        self.metric_category = MetricCategory.SOUNDNESS
        self.validate_explainer()

    def __call__(self, multiclass=False):
        """
        Calculates the fidelity metric between the original model predictions and the explainer predictions.

        Parameters:
            multiclass (bool): Flag indicating whether the fidelity metric should be calculated for multiclass classification. 
                            If False, the binary F1 score is calculated. If True, the weighted F1 score is calculated.

        Returns:
            float: The fidelity metric score.
        """
        original_model_labels = self.dataset.y_predicted
        explainer_probabilities = self.explainer.explainer.predict_proba(
            self.dataset.X_vectorized)

        # Convert probabilities to class labels for the explainer predictions
        explainer_labels = np.argmax(explainer_probabilities, axis=1)

        # Calculate and return the F1 score for fidelity
        avg = 'binary' if not multiclass else 'weighted'
        fidelity = f1_score(original_model_labels,
                            explainer_labels, average=avg)
        return fidelity
