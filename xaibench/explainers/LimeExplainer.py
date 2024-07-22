from .Explainer import Explainer
import logging
from utilities import ExplanationType, ExplanationScope

from lime.lime_text import LimeTextExplainer

logger = logging.getLogger(__name__)


class LimeExplainer(Explainer):
    """
    A specialized explainer utilizing the LIME framework to interpret text data predictions.

    Inherits from the Explainer class, focusing on providing explanations for predictions made on text data.
    Utilizes the LimeTextExplainer for creating interpretable and local explanations for why models make certain predictions.

    Attributes:
        dataset: A dataset object that includes a preprocessing pipeline and class names necessary for explanations.
        explainer: An instance of LimeTextExplainer, initialized to work with the specific dataset's class names.
    """

    def __init__(self, dataset, **kwargs):
        """
        Initializes the LimeExplainer with a dataset and optionally additional keyword arguments for the parent class.

        Args:
            dataset: The dataset used for training the model. It should provide a preprocessing pipeline via
                     `dataset.pipe` and a list of class names via `dataset.class_names`.
            **kwargs: Additional keyword arguments that are passed to the superclass initializer.
        """
        super().__init__(**kwargs)
        self.dataset = dataset
        self.scope = ExplanationScope.LOCAL
        self.explanation_type = ExplanationType.FEATURE
        self.explainer = LimeTextExplainer(class_names=dataset.class_names)

    def __call__(self, instance, n_features=6):
        """
        Generates an explanation for a given instance of text, specifying how much each part of the text contributed
        to the model's prediction.

        This method uses LIME to explain the prediction for a single instance of text, returning a list of features
        (words or phrases) and their contributions to the prediction.

        Args:
            instance: The text instance to explain.
            n_features: The number of features to include in the explanation. Defaults to 6.

        Returns:
            A list of tuples, each representing a feature and its contribution to the prediction, allowing for an
            understanding of which parts of the text were most influential.
        """
        exp = self.explainer.explain_instance(
            instance, self.dataset.pipe.predict_proba, num_features=n_features)
        return exp.as_list()
