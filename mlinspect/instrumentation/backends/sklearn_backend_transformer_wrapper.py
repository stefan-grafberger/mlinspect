"""
A wrapper for sklearn transformers to capture method calls we do not see otherwise because of the pipeline
definition style
"""
import inspect

from sklearn.base import BaseEstimator


class MlinspectEstimatorTransformer(BaseEstimator):
    """
    A wrapper for sklearn transformers to capture method calls we do not see otherwise because of the pipeline
    definition style
    See: https://scikit-learn.org/stable/developers/develop.html
    """

    def __init__(self, transformer):
        self.transformer = transformer
        self.name = transformer.__class__.__name__
        self.module_info = inspect.getmodule(transformer)

    def fit(self, X: list, y=None) -> 'MlinspectEstimatorTransformer':
        """
        Override fit
        """
        # pylint: disable=invalid-name
        print(self.name)
        print("fit:")
        print("X")
        print(X)
        print("y")
        print(y)
        self.transformer = self.transformer.fit(X, y)
        return self

    def transform(self, X: list) -> list:
        """
        Override transform
        """
        # pylint: disable=invalid-name
        print(self.name)
        print("transform:")
        print("X")
        print(X)
        result = self.transformer.transform(X)
        print("result")
        print(result)
        return result

    def fit_transform(self, X: list, y=None) -> list:  # TODO: There can be some additional kwargs sometimes
        """
        Override fit_transform
        """
        # pylint: disable=invalid-name
        print(self.name)
        print("fit_transform:")
        print("X")
        print(X)
        print("y")
        print(y)
        result = self.transformer.fit_transform(X, y)
        print("result")
        print(result)
        return result
