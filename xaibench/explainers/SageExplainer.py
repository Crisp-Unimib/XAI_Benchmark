# from .Explainer import Explainer

from .Explainer import Explainer
import logging

from utilities import ExplanationType, ExplanationScope
import sage

import numpy as np

logger = logging.getLogger(__name__)


class SageWrapper():
    def __init__(self, dataset, maxDocuments, loss):
        # The explainer is initialized during the __call__() execution because
        # the constructor needs some x data
        self.maxDocuments = maxDocuments
        self.loss = loss
        self.feature_names = dataset.feature_names

        # It is only the calibrated classifier, it is not a pipeline
        self.model = dataset.model

    def fit(self, x, y, **kwargs):
        x = x.toarray()  # .todense() will break the next code
        if type(y) != type(np.array([1])):
            print('******')
            y = y.to_numpy()
        # The work will focus on the first {nItems} doucments
        if len(x) > self.maxDocuments:
            x = x[:self.maxDocuments, :].copy()
            y = y[:self.maxDocuments].copy().astype(int)
        else:
            x = x.copy()
            y = y.copy().astype(int)

        imputer = sage.MarginalImputer(self.model, x)
        # Supported loss: 'mse', 'cross entropy'
        self.explainer = sage.PermutationEstimator(
            imputer, self.loss, random_state=42)
        return self.explainer(x, y)

    def predict(self, instances):
        pass

    def predict_proba(self, instances):
        pass


class SageExplainer(Explainer):
    """
    A specialized explainer utilizing the SAGE framework to interpret text data predictions.

    Inherits from the Explainer class, focusing on providing explanations for predictions made on text data.
    Utilizes a combination of 
    * marginalizing-out features 
    * loss computation to identify the features whose exclusion make the loss worsening
    * shape values to quantify the effect of each feature
    Sage uses a global explaination approach. .


    """

    def __init__(self, dataset, maxDocuments=100, loss='cross entropy', **kwargs):
        """
        Initializes the ShapExplainer with a machine learning model and a tokenizer for preprocessing text.

        Parameters:
            dataset: An object containing 
                     * The dataset used for training the model. It must have 
                       `X_vectorized` (pandas.DataFrame or similar structure) 
                     * A classification/preprocessing pipeline 
                     * A vectorizer object which will be used to get the feature names
                     * `y_predicted` (pandas.Series or similar structure)
                     as the target labels.

            Parameters:
                maxDocuments: the max number of documents that will be used 
                                by the explainer

                loss: The loss to be used with Sage. Possible values are 'mse' and 'cross entropy'

            **kwargs: Additional keyword arguments for further customization.
        """
        super().__init__(**kwargs)
        self.dataset = dataset
        self.scope = ExplanationScope.GLOBAL  # ExplanationScope.LOCAL
        self.explanation_type = ExplanationType.FEATURE  # ExplanationType.RULE

        # Calculate SAGE values
        self.explainer = SageWrapper(dataset, maxDocuments, loss)
        self.sage_values = self.explainer.fit(
            self.dataset.X_vectorized, self.dataset.y_predicted)

    def __call__(self, n_features=6):
        """
        Generates a global explanation for the given dataset, 
        specifying the impact of each part of the text on the model's prediction.

        This method uses SAGE to get a global explanation
        returning the contribution of each feature to the prediction.

        Parameters:

            n_features: The number of features to include in the explanation. 
                        Defaults to 6.
        """

        vals = self.sage_values.values
        # ci sarebbe anche la std, ma non la ho inserita nell'output sottostante
        # variance = sage_values.std

        feature_importances = []
        for feature, val in zip(self.dataset.feature_names, vals):
            feature_importances.append((feature, val))

        feature_importances = sorted(
            feature_importances, key=lambda x: x[1], reverse=True)
        # Return only the top n_features values
        return feature_importances[:n_features]
