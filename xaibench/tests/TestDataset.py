import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
import scipy
from data.Imdb import Imdb


class TestImdb(unittest.TestCase):
    def setUp(self):
        self.dataset = Imdb(seed=123, n_features=10)

    def test_init(self):
        """
        Test the initialization of the dataset.

        This method checks if the dataset is initialized correctly by
        asserting the values of the seed and the number of features.

        """
        self.assertEqual(self.dataset.seed, 123)
        self.assertEqual(self.dataset.n_features, 10)

    def test_dataframe(self):
        """
        Test if the dataset's df attribute is a pandas DataFrame and contains the expected columns.
        """
        self.assertIsInstance(self.dataset.df, pd.DataFrame)
        self.assertIn('text', self.dataset.df.columns)
        self.assertIn('target', self.dataset.df.columns)

    def test_X(self):
        self.assertIsInstance(self.dataset.X, pd.Series)

    def test_y(self):
        self.assertIsInstance(self.dataset.y, pd.Series)

    def test_y_predicted(self):
        self.assertIsInstance(self.dataset.y_predicted, np.ndarray)

    def test_feature_names(self):
        self.assertIsNotNone(self.dataset.feature_names)

    def test_model(self):
        self.assertIsNotNone(self.dataset.model)

    def test_vectorizer(self):
        self.assertIsNotNone(self.dataset.vectorizer)

    def test_class_names(self):
        """
        Test case to verify the class names in the dataset.

        It checks if the `class_names` attribute is a list, has a length of 2,
        and contains the expected class names 'positive' and 'negative'.
        """
        self.assertIsInstance(self.dataset.class_names, list)
        self.assertEqual(len(self.dataset.class_names), 2)
        self.assertIn('positive', self.dataset.class_names)
        self.assertIn('negative', self.dataset.class_names)

    def test_vectorized_data(self):
        self.assertIsInstance(self.dataset.X_vectorized,
                              scipy.sparse.csr_matrix)

    def test_pipeline(self):
        self.assertIsNotNone(self.dataset.pipe)

    @patch('pickle.load')
    def test_model_loading(self, mock_pickle_load):
        """
        Test the loading of a model using mock_pickle_load.

        Args:
            mock_pickle_load: The mock object for pickle.load.

        Returns:
            None
        """
        mock_pickle_load.return_value = Mock()
        _ = Imdb(seed=123, n_features=10)
        self.assertTrue(mock_pickle_load.called)

    @patch('pickle.load')
    def test_vectorizer_loading(self, mock_pickle_load):
        """
        Test case to verify if the vectorizer is loaded correctly.

        Args:
            mock_pickle_load: Mock object for pickle.load().

        Returns:
            None
        """
        mock_pickle_load.return_value = Mock()
        _ = Imdb(seed=123, n_features=10)
        self.assertTrue(mock_pickle_load.called)


if __name__ == '__main__':
    unittest.main()
