from .Explainer import Explainer
from utilities import ExplanationType, ExplanationScope
import logging

from anchor import anchor_text
import numpy as np
import spacy

logger = logging.getLogger(__name__)


class AnchorExplainer(Explainer):
    """
    An explainer for text data using the Anchors method to provide interpretable explanations
    for model predictions.
    """

    def __init__(self, dataset, **kwargs):
        """
        Initializes the AnchorsTextExplainer.

        Parameters:
            dataset: The dataset used for training the model, expected to have
                     X_vectorized (features) and y_predicted (target labels).
        """
        super().__init__(**kwargs)
        self.dataset = dataset
        self.scope = ExplanationScope.LOCAL
        self.explanation_type = ExplanationType.RULE
        # Load the spaCy model for text processing
        # python -m spacy download en_core_web_lg
        nlp = spacy.load('en_core_web_sm')
        self.explainer = anchor_text.AnchorText(
            nlp=nlp, class_names=self.dataset.class_names, use_unk_distribution=True)

    def __call__(self, instance, n_rules=6, threshold=0.95):
        """
        Generates an explanation for a single instance.

        Parameters:
            instance (str): The text instance to explain.
            threshold (float): The precision threshold for the explanation.

        Returns:
            The explanation object, which includes the anchor.
        """
        if not n_rules or n_rules < 1:
            n_rules = 1000

        explanation = self.explainer.explain_instance(
            instance,
            self.dataset.pipe.predict,
            verbose=False,
            threshold=threshold,
            onepass=True,
        )
        rules = [{f: ('>', 0) for f in explanation.names()}]
        return rules[:n_rules]
