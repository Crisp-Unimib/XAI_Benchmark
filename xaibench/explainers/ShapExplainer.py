from .Explainer import Explainer
import logging
from utilities import ExplanationType, ExplanationScope

import shap
import numpy as np

logger = logging.getLogger(__name__)


class ShapExplainer(Explainer):
    """
    A specialized explainer utilizing the SHAP framework to interpret text data predictions.

    Inherits from the Explainer class, focusing on providing explanations for predictions made on text data.
    Utilizes SHAP, a game theory approach to explain the output of any machine learning model, for creating interpretable
    and local explanations for why models make certain predictions.

    Attributes:
        dataset: A dataset object that includes a preprocessing pipeline and class names necessary for explanations.
        explainer: An instance of a SHAP explainer, initialized to work with the specified model.
    """

    def __init__(self, dataset, **kwargs):
        """
        Initializes the ShapExplainer with a machine learning model and a tokenizer for preprocessing text.

        Args:
            dataset: A dataset object that includes a preprocessing pipeline and class names necessary for explanations.
            **kwargs: Additional keyword arguments that are passed to the superclass initializer.
        """
        super().__init__(**kwargs)
        self.dataset = dataset
        self.scope = ExplanationScope.LOCAL
        self.explanation_type = ExplanationType.FEATURE
        # Initialize a SHAP explainer. KernelExplainer or DeepExplainer are commonly used with text models,
        # depending on the model type.
        self.explainer = shap.KernelExplainer(
            dataset.model.predict_proba, data=shap.sample(dataset.X_vectorized,
                                                          nsamples=max(100, self.dataset.n_features)), silent=True)

    def __call__(self, instance, n_features=6):
        """
        Generates an explanation for a given instance of text, specifying the impact of each part of the text on the model's prediction.

        This method uses SHAP to explain the prediction for a single instance of text, returning a SHAP values object that
        indicates the contribution of each feature to the prediction.

        Args:
            instance: The text instance to explain, in raw string format.
            n_features: The number of features to include in the explanation. Defaults to 6.

        Returns:
            A SHAP values object representing the contribution of each feature (part of the text) to the model's prediction.
            This object can be further used to generate plots or detailed explanations.
        """
        # If the instance is a string, vectorize it using the dataset's vectorizer
        if isinstance(instance, str):
            instance = self.dataset.vectorizer.transform([instance])
        # Generate SHAP values for the text
        shap_values = self.explainer.shap_values(
            instance, nsamples=self.dataset.n_features, silent=True)
        # Create a list of tuples containing feature names and their SHAP values
        vals = np.abs(shap_values).mean(0)[0]
        feature_importances = []
        for feature, val in zip(self.dataset.feature_names, vals):
            feature_importances.append((feature, val))
        # Sort the feature importances by their SHAP values
        feature_importances = sorted(
            feature_importances, key=lambda x: x[1], reverse=True)
        # Return only the top n_features SHAP values for simplicity
        return feature_importances[:n_features]
