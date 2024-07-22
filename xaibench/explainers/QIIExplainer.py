from .Explainer import Explainer
import logging
from utilities import ExplanationType, ExplanationScope
import numpy as np

from qii.qii import QII
from qii.predictor import QIIPredictor
from qii.qoi import QuantityOfInterest

logger = logging.getLogger(__name__)


class Predictor(QIIPredictor):
    def __init__(self, predictor):
        super(Predictor, self).__init__(predictor)

    def predict(self, x):
        # predict the label for instance x
        return self._predictor.predict(x)


class QIIExplainer(Explainer):

    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)
        self.scope = ExplanationScope.LOCAL
        self.explanation_type = ExplanationType.FEATURE
        self.dataset = dataset

        self._predictor = Predictor(dataset.model)
        quantity_of_interest = QuantityOfInterest()
        self.explainer = QII(np.array(self.dataset.X_vectorized.todense(
        )), self.dataset.n_features, quantity_of_interest)

    def __call__(self, instance=None, n_features=6):
        if instance is None:
            instance = self.dataset.X[0]
        instance = self.dataset.vectorizer.transform([instance])
        instance = np.array(instance.todense())

        # Compute Shapley values
        banzhaf_vals = self.explainer.compute(x_0=instance, predictor=self._predictor,
                                              show_approx=True, evaluated_features=None,
                                              data_exhaustive=False, feature_exhaustive=False,
                                              method='banzhaf')
        # Replace feature indices with feature names
        banzhaf_vals = {self.dataset.feature_names[i]: banzhaf_vals[i] for i in range(
            len(banzhaf_vals))}
        # Sort by feature importance
        banzhaf_vals = sorted(banzhaf_vals.items(),
                              key=lambda x: x[1], reverse=True)
        return banzhaf_vals[:n_features]
