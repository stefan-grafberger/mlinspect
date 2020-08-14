"""
A wrapper for sklearn transformers to capture method calls we do not see otherwise because of the pipeline
definition style
"""
import inspect

from sklearn.base import BaseEstimator

from mlinspect.instrumentation.dag_node import CodeReference


class MlinspectEstimatorTransformer(BaseEstimator):
    """
    A wrapper for sklearn transformers to capture method calls we do not see otherwise because of the pipeline
    definition style
    See: https://scikit-learn.org/stable/developers/develop.html
    """

    def __init__(self, transformer, code_reference: CodeReference):
        self.transformer = transformer
        self.name = transformer.__class__.__name__

        module = inspect.getmodule(transformer)
        self.module_name = module.__name__
        self.call_function_info = (module.__name__, transformer.__class__.__name__)
        self.code_reference = code_reference

    def fit(self, X: list, y=None) -> 'MlinspectEstimatorTransformer':
        """
        Override fit
        """
        # pylint: disable=invalid-name
        if self.call_function_info == ('sklearn.pipeline', 'Pipeline'):
            print("pipeline")
            # TODO: returns nothing but allows a scan for train data and train labels. output is same as input
        print(self.call_function_info[1])
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
        print(self.call_function_info[1])
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
        if self.call_function_info == ('sklearn.compose._column_transformer', 'ColumnTransformer'):
            print("column transformer")

        print(self.call_function_info[1])
        print("fit_transform:")
        print("X")
        print(X)
        print("y")
        print(y)
        result = self.transformer.fit_transform(X, y)
        print("result")
        print(result)
        return result
