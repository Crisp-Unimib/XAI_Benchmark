from . import Dataset
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
import pickle
import logging

logger = logging.getLogger(__name__)


class Imdb(Dataset):
    """
    A class for handling the IMDb movie review dataset for sentiment analysis tasks. It automates
    the process of loading the dataset, vectorizing text reviews, and preparing data for model training
    or prediction using a pre-trained model.

    Attributes:
        df (pandas.DataFrame): DataFrame containing the loaded IMDb dataset with `text` and `target` columns.
        X (pandas.Series): Series containing the text reviews.
        y (pandas.Series): Series containing the sentiment labels.
        y_predicted (pandas.DataFrame): DataFrame containing the predicted sentiments for the reviews.
        class_names: The class names for the sentiment labels.
        model: The pre-trained sentiment analysis model loaded from a pickle file.
        vectorizer (CountVectorizer): Vectorizer instance for converting text reviews to vectorized format.
        X_vectorized: The vectorized representation of `X`.
        n_features (int): Maximum number of features for vectorization.
        pipe (Pipeline): A scikit-learn pipeline combining `vectorizer` and `model`.

    Methods:
        __init__(self, seed=42, n_features=None, **kwargs):
            Initializes the Imdb class with dataset, model, and vectorizer setup.

        __call__(self, **kwargs): Returns the IMDb dataset DataFrame.
        __repr__(self): Returns the canonical string representation of the Imdb class.
        __str__(self): Returns a more human-readable string representation of the Imdb class.
    """

    def __init__(self, seed=42, n_features=10, **kwargs):
        """
        Initializes the Imdb class with a specified random seed for reproducibility, an optional
        number of features for vectorization, and additional keyword arguments for the Dataset superclass.

        Parameters:
            seed (int): Seed for random number generation to ensure reproducibility. Defaults to 42.
            n_features (int, optional): Maximum number of features to use for vectorization. Defaults to None.
            **kwargs: Additional keyword arguments passed to the Dataset superclass.
        """
        super().__init__(seed, **kwargs)

        self.df = pd.read_csv('data/imdb.csv')
        self.X = self.df['text']
        self.y = self.df['target']
        self.class_names = ['negative', 'positive']

        model_params_filename = f'models/predictions/svc_imdb_{str(n_features)}.sav'
        vectorizer_filename = f'models/vectorizer/vectorizer_imdb_{str(n_features)}.pkl'
        with open(model_params_filename, 'rb') as file:
            self.model = pickle.load(file)
        with open(vectorizer_filename, 'rb') as file:
            self.vectorizer = pickle.load(file)
        self.X_vectorized = self.vectorizer.transform(self.X)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.n_features = n_features
        self.pipe = make_pipeline(self.vectorizer, self.model)
        self.y_predicted = self.pipe.predict(self.X)
        logger.info(f'Imdb dataset loaded with {len(self.df)} samples.')

    def __call__(self, **kwargs):
        """
        Allows the Imdb instance to be called like a function, returning the IMDb dataset DataFrame.

        Parameters:
            **kwargs: Additional keyword arguments (unused).

        Returns:
            pandas.DataFrame: The DataFrame containing the IMDb dataset.
        """
        return self.df

    def __repr__(self):
        """Returns the canonical string representation of the Imdb class."""
        return "Imdb"

    def __str__(self):
        """Returns a more human-readable string representation of the Imdb class."""
        return "Imdb"
