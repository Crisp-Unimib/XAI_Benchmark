from .Metric import Metric
import logging

logger = logging.getLogger(__name__)


class StressFeatureMetric(Metric):
    """
    https://github.com/Karim-53/Compare-xAI/blob/main/tests/stress_nb_features.py
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self):
        pass
