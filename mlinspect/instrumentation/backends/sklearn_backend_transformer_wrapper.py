"""
A wrapper for sklearn transformers to capture method calls we do not see otherwise because of the pipeline
definition style
"""
import inspect
import itertools
from functools import partial

import numpy
from pandas import DataFrame
from sklearn.base import BaseEstimator

from mlinspect.instrumentation.analyzers.analyzer_input import OperatorContext, AnalyzerInputRow, \
    AnalyzerInputUnaryOperator
from mlinspect.instrumentation.backends.pandas_backend_frame_wrapper import MlinspectDataFrame
from mlinspect.instrumentation.backends.sklearn_backend_ndarray_wrapper import MlinspectNdarray
from mlinspect.instrumentation.dag_node import CodeReference, OperatorType


class MlinspectEstimatorTransformer(BaseEstimator):
    """
    A wrapper for sklearn transformers to capture method calls we do not see otherwise because of the pipeline
    definition style
    See: https://scikit-learn.org/stable/developers/develop.html
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, transformer, code_reference: CodeReference, analyzers, code_reference_analyzer_output_map,
                 output_dimensions=None, column_transformer_annotations=None):
        self.transformer = transformer
        self.name = transformer.__class__.__name__

        module = inspect.getmodule(transformer)
        self.module_name = module.__name__
        self.call_function_info = (module.__name__, transformer.__class__.__name__)
        self.code_reference = code_reference
        self.analyzers = analyzers
        self.code_reference_analyzer_output_map = code_reference_analyzer_output_map
        self.output_dimensions = output_dimensions
        self.column_transformer_annotations = column_transformer_annotations

    def fit(self, X: list, y=None) -> 'MlinspectEstimatorTransformer':
        """
        Override fit
        """
        # pylint: disable=invalid-name
        if self.call_function_info == ('sklearn.pipeline', 'Pipeline'):
            # Pipeline.fit returns nothing (trained models get no edge in our DAG)
            # Only need to do two scans for train data and train labels
            function_info = (self.module_name, "fit")  # TODO: nested pipelines
            operator_context = OperatorContext(OperatorType.TRAIN_DATA, function_info)
            execute_analyzer_visits_df_input_df_output(operator_context, self.code_reference, X, X, self.analyzers,
                                                       self.code_reference_analyzer_output_map, "fit X")
            if y is not None:
                assert isinstance(y, MlinspectNdarray)
                operator_context = OperatorContext(OperatorType.TRAIN_LABELS, function_info)
                execute_analyzer_visits_array_input_array_output(operator_context, self.code_reference, y, y,
                                                                 self.analyzers,
                                                                 self.code_reference_analyzer_output_map, "fit y")
        elif self.call_function_info == ('sklearn.tree._classes', 'DecisionTreeClassifier'):
            print("DecisionTreeClassifier")
        print(self.call_function_info[1])
        print("fit:")
        print("X")
        print(X)
        print("y")
        print(y)
        self.transformer = self.transformer.fit(X, y)
        print("analyzer output:")
        print(self.code_reference_analyzer_output_map)
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
            # Analyzers for the different projections
            transformers_tuples = self.transformer.transformers
            columns_with_transformer = [(column, transformer_tuple[1]) for transformer_tuple in transformers_tuples
                                        for column in transformer_tuple[2]]
            for column, transformer in columns_with_transformer:
                projected_df = X[[column]]
                function_info = (self.module_name, "fit_transform")  # TODO: nested pipelines
                operator_context = OperatorContext(OperatorType.PROJECTION, function_info)
                description = "to ['{}']".format(column)
                execute_analyzer_visits_df_input_df_output(operator_context, self.code_reference, projected_df,
                                                           projected_df, self.analyzers,
                                                           self.code_reference_analyzer_output_map, description,
                                                           transformer)
            # ---
            result = self.transformer.fit_transform(X, y)
            # ---
            # Analyzers for concat, use the self.output dimensions attribute to associate result columns
        elif self.call_function_info == ('sklearn.preprocessing._encoders', 'OneHotEncoder'):
            assert self.column_transformer_annotations is not None
            print("OneHotEncoder")
            result = self.transformer.fit_transform(X, y)
            self.output_dimensions = [len(one_hot_categories) for one_hot_categories in self.transformer.categories_]
        elif self.call_function_info == ('sklearn.preprocessing._data', 'StandardScaler'):
            assert self.column_transformer_annotations is not None
            result = self.transformer.fit_transform(X, y)
            print("StandardScaler")
            self.output_dimensions = [1 for _ in range(result.shape[1])]
        else:
            result = self.transformer.fit_transform(X, y)

        print(self.call_function_info[1])
        print("fit_transform:")
        print("X")
        print(X)
        print("y")
        print(y)

        print("result")
        print(result)
        return result


def execute_analyzer_visits_df_input_df_output(operator_context, code_reference, input_data, output_data, analyzers,
                                               code_reference_analyzer_output_map, func_name, transformer=None):
    """Execute analyzers when the current operator has one parent in the DAG"""
    # pylint: disable=too-many-arguments
    assert isinstance(input_data, MlinspectDataFrame)
    annotation_iterators = []
    for analyzer in analyzers:
        analyzer_index = analyzers.index(analyzer)
        iterator_for_analyzer = iter_input_annotation_output_df_df(analyzer_index,
                                                                   input_data,
                                                                   input_data.annotations,
                                                                   output_data)
        annotations_iterator = analyzer.visit_operator(operator_context, iterator_for_analyzer)
        annotation_iterators.append(annotations_iterator)
    return_value = store_analyzer_outputs_df(annotation_iterators, code_reference, output_data, analyzers,
                                             code_reference_analyzer_output_map, func_name, transformer)
    assert isinstance(return_value, MlinspectDataFrame)
    return return_value


def execute_analyzer_visits_array_input_array_output(operator_context, code_reference, input_data, output_data,
                                                     analyzers, code_reference_analyzer_output_map, func_name):
    """Execute analyzers when the current operator has one parent in the DAG"""
    # pylint: disable=too-many-arguments
    assert isinstance(input_data, MlinspectNdarray)
    annotation_iterators = []
    for analyzer in analyzers:
        analyzer_index = analyzers.index(analyzer)
        iterator_for_analyzer = iter_input_annotation_output_array_array(analyzer_index,
                                                                         input_data,
                                                                         input_data.annotations,
                                                                         output_data)
        annotations_iterator = analyzer.visit_operator(operator_context, iterator_for_analyzer)
        annotation_iterators.append(annotations_iterator)
    return_value = store_analyzer_outputs_array(annotation_iterators, code_reference, output_data, analyzers,
                                                code_reference_analyzer_output_map, func_name)
    assert isinstance(return_value, MlinspectNdarray)
    return return_value


def iter_input_annotation_output_df_df(analyzer_index, input_df, annotation_df, output_df):
    """
    Create an efficient iterator for the analyzer input for operators with one parent.
    """
    # pylint: disable=too-many-locals
    # Performance tips:
    # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
    assert isinstance(input_df, DataFrame)
    assert isinstance(output_df, DataFrame)

    annotation_df_view = annotation_df.iloc[:, analyzer_index:analyzer_index + 1]

    input_rows = get_df_row_iterator(input_df)
    annotation_rows = get_df_row_iterator(annotation_df_view)
    output_rows = get_df_row_iterator(output_df)

    return map(lambda input_tuple: AnalyzerInputUnaryOperator(*input_tuple),
               zip(input_rows, annotation_rows, output_rows))


def iter_input_annotation_output_array_array(analyzer_index, input_df, annotation_df, output_df):
    """
    Create an efficient iterator for the analyzer input for operators with one parent.
    """
    # pylint: disable=too-many-locals
    # Performance tips:
    # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
    assert isinstance(input_df, numpy.ndarray)
    assert isinstance(output_df, numpy.ndarray)

    annotation_df_view = annotation_df.iloc[:, analyzer_index:analyzer_index + 1]

    input_rows = get_numpy_array_row_iterator(input_df)
    annotation_rows = get_df_row_iterator(annotation_df_view)
    output_rows = get_numpy_array_row_iterator(output_df)

    return map(lambda input_tuple: AnalyzerInputUnaryOperator(*input_tuple),
               zip(input_rows, annotation_rows, output_rows))


def get_df_row_iterator(dataframe):
    """
    Create an efficient iterator for the data frame rows.
    The implementation is inspired by the implementation of the pandas DataFrame.itertuple method
    """
    arrays = []
    fields = list(dataframe.columns)
    arrays.extend(dataframe.iloc[:, k] for k in range(0, len(dataframe.columns)))

    partial_func_create_row = partial(AnalyzerInputRow, fields=fields)
    test = map(partial_func_create_row, map(list, zip(*arrays)))
    return test


def get_numpy_array_row_iterator(nparray):
    """
    Create an efficient iterator for the data frame rows.
    The implementation is inspired by the implementation of the pandas DataFrame.itertuple method
    """
    fields = list(["array"])
    numpy_iterator = numpy.nditer(nparray, ["refs_ok"])
    partial_func_create_row = partial(AnalyzerInputRow, fields=fields)

    test = map(partial_func_create_row, map(list, zip(numpy_iterator)))
    return test


def store_analyzer_outputs_df(annotation_iterators, code_reference, return_value, analyzers,
                              code_reference_analyzer_output_map, func_name, transformer=None):
    """
    Stores the analyzer annotations for the rows in the dataframe and the
    analyzer annotations for the DAG operators in a map
    """
    # pylint: disable=too-many-arguments
    annotation_iterators = itertools.zip_longest(*annotation_iterators)
    analyzer_names = [str(analyzer) for analyzer in analyzers]
    annotations_df = DataFrame(annotation_iterators, columns=analyzer_names)
    analyzer_outputs = {}
    for analyzer in analyzers:
        analyzer_output = analyzer.get_operator_annotation_after_visit()
        analyzer_outputs[analyzer] = analyzer_output

    stored_analyzer_results = code_reference_analyzer_output_map.get(code_reference, {})
    stored_analyzer_results[func_name] = analyzer_outputs
    code_reference_analyzer_output_map[code_reference] = stored_analyzer_results
    return_value = MlinspectDataFrame(return_value)

    # if the transformer is a column transformer, we have multiple annotations we need to pass to different transformers
    # if we do not want to override internal column transformer functions, we have to work around these black
    # box functions and pass the annotations using a different mechanism
    if transformer is None:
        return_value.annotations = annotations_df
    else:
        return_value.annotations = None
        previous_annotations = transformer.column_transformer_annotations or []
        previous_annotations.append(annotations_df)
        transformer.column_transformer_annotations = previous_annotations
    assert isinstance(return_value, MlinspectDataFrame)
    return return_value


def store_analyzer_outputs_array(annotation_iterators, code_reference, return_value, analyzers,
                                 code_reference_analyzer_output_map, func_name):
    """
    Stores the analyzer annotations for the rows in the dataframe and the
    analyzer annotations for the DAG operators in a map
    """
    # pylint: disable=too-many-arguments
    annotation_iterators = itertools.zip_longest(*annotation_iterators)
    analyzer_names = [str(analyzer) for analyzer in analyzers]
    annotations_df = DataFrame(annotation_iterators, columns=analyzer_names)
    analyzer_outputs = {}
    for analyzer in analyzers:
        analyzer_output = analyzer.get_operator_annotation_after_visit()
        analyzer_outputs[analyzer] = analyzer_output

    stored_analyzer_results = code_reference_analyzer_output_map.get(code_reference, {})
    stored_analyzer_results[func_name] = analyzer_outputs
    code_reference_analyzer_output_map[code_reference] = stored_analyzer_results
    return_value = MlinspectNdarray(return_value)
    return_value.annotations = annotations_df
    assert isinstance(return_value, MlinspectNdarray)
    return return_value
