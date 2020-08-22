"""
A wrapper for sklearn transformers to capture method calls we do not see otherwise because of the pipeline
definition style
"""
import inspect
import uuid

import numpy
from pandas import DataFrame
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator

from mlinspect.instrumentation.analyzers.analyzer_input import OperatorContext, AnalyzerInputUnaryOperator, \
    AnalyzerInputNAryOperator, AnalyzerInputSinkOperator
from mlinspect.instrumentation.backends.backend_utils import get_numpy_array_row_iterator, get_df_row_iterator, \
    get_csr_row_iterator, build_annotation_df_from_iters, get_series_row_iterator
from mlinspect.instrumentation.backends.pandas_backend_frame_wrapper import MlinspectDataFrame, MlinspectSeries
from mlinspect.instrumentation.backends.sklearn_backend_csr_matrx_wrapper import MlinspectCsrMatrix
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
                 output_dimensions=None, transformer_uuid=None):
        # pylint: disable=too-many-arguments
        # None arguments are not passed directly when we create them. Still needed though because the
        # Column transformer clones child transformers and does not pass parameters otherwise
        self.transformer = transformer
        self.name = transformer.__class__.__name__

        module = inspect.getmodule(transformer)
        self.module_name = module.__name__
        self.call_function_info = (module.__name__, transformer.__class__.__name__)
        self.code_reference = code_reference
        self.analyzers = analyzers
        self.code_reference_analyzer_output_map = code_reference_analyzer_output_map
        self.output_dimensions = output_dimensions
        self.annotation_result_concat_workaround = None
        if transformer_uuid is None:
            self.transformer_uuid = uuid.uuid4()
        else:
            self.transformer_uuid = transformer_uuid

    def __eq__(self, other):
        return isinstance(other, MlinspectEstimatorTransformer) and self.transformer_uuid == other.transformer_uuid

    def __hash__(self):
        return hash(self.transformer_uuid)

    def fit(self, X, y=None) -> 'MlinspectEstimatorTransformer':
        """
        Override fit
        """
        # pylint: disable=invalid-name
        if self.call_function_info == ('sklearn.pipeline', 'Pipeline'):
            X_new, y_new = self.train_data_and_labels_visits(X, y)
        elif self.call_function_info in {('sklearn.tree._classes', 'DecisionTreeClassifier'),
                                         ('tensorflow.python.keras.wrappers.scikit_learn', 'KerasClassifier')}:
            X_new, y_new = self.estimator_visits(X, y)
        else:
            X_new = X
            y_new = y
        self.transformer = self.transformer.fit(X_new, y_new)
        return self

    def transform(self, X: list) -> list:
        """
        Override transform
        """
        # pylint: disable=invalid-name
        result = self.transformer.transform(X)
        return result

    def fit_transform(self, X, y=None) -> list:  # TODO: There can be some additional kwargs sometimes
        """
        Override fit_transform
        """
        # pylint: disable=invalid-name
        if self.call_function_info == ('sklearn.compose._column_transformer', 'ColumnTransformer'):
            result = self.column_transformer_visits(X, y)
        elif self.call_function_info == ('sklearn.preprocessing._encoders', 'OneHotEncoder'):
            result = self.one_hot_encoder_visits(X, y)
        elif self.call_function_info == ('sklearn.preprocessing._data', 'StandardScaler'):
            result = self.standard_scaler_visits(X, y)
        elif self.call_function_info == ('mlinspect.demo_utils', 'MyW2VTransformer'):
            result = self.w2v_visits(X, y)
        else:
            result = self.transformer.fit_transform(X, y)

        return result

    def standard_scaler_visits(self, X, y):
        """
        Analyzer visits for the StandardScaler Transformer
        """
        # pylint: disable=invalid-name
        assert isinstance(X.annotations, dict) and self in X.annotations
        result = self.transformer.fit_transform(X, y)
        self.output_dimensions = [1 for _ in range(result.shape[1])]
        for column_index, column in enumerate(X.columns):
            function_info = (self.module_name, "fit_transform")  # TODO: nested pipelines
            operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
            description = "Numerical Encoder (StandardScaler), Column: '{}'".format(column)
            column_result = execute_analyzer_visits_df_array_column_transformer(operator_context,
                                                                                self.code_reference,
                                                                                X[[column]],
                                                                                X.annotations[self],
                                                                                result[:, column_index],
                                                                                self.analyzers,
                                                                                self.code_reference_analyzer_output_map,
                                                                                description)
            annotations_for_columns = self.annotation_result_concat_workaround or []
            annotations_for_columns.append(column_result.annotations)
            self.annotation_result_concat_workaround = annotations_for_columns
        return result

    def w2v_visits(self, X, y):
        """
        Analyzer visits for the StandardScaler Transformer
        """
        # pylint: disable=invalid-name
        assert isinstance(X.annotations, dict) and self in X.annotations
        result = self.transformer.fit_transform(X, y)
        self.output_dimensions = [result.shape[1]]
        for column_index, column in enumerate(X.columns):
            function_info = (self.module_name, "fit_transform")  # TODO: could also be used for multiple columns at once
            operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
            description = "Word2Vec, Column: '{}'".format(column)
            column_result = execute_analyzer_visits_df_array_column_transformer(operator_context,
                                                                                self.code_reference,
                                                                                X[[column]],
                                                                                X.annotations[self],
                                                                                result[:, :],
                                                                                self.analyzers,
                                                                                self.code_reference_analyzer_output_map,
                                                                                description)
            annotations_for_columns = self.annotation_result_concat_workaround or []
            annotations_for_columns.append(column_result.annotations)
            self.annotation_result_concat_workaround = annotations_for_columns
        return result

    def one_hot_encoder_visits(self, X, y):
        """
        Analyzer visits for the OneHotEncoder Transformer
        """
        # pylint: disable=invalid-name
        assert isinstance(X.annotations, dict) and self in X.annotations
        result = self.transformer.fit_transform(X, y)
        self.output_dimensions = [len(one_hot_categories) for one_hot_categories in
                                  self.transformer.categories_]
        output_dimension_index = [0]
        for dimension in self.output_dimensions:
            output_dimension_index.append(output_dimension_index[-1] + dimension)
        for column_index, column in enumerate(X.columns):
            function_info = (self.module_name, "fit_transform")  # TODO: nested pipelines
            operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
            description = "Categorical Encoder (OneHotEncoder), Column: '{}'".format(column)

            index_start = output_dimension_index[column_index]
            index_end = output_dimension_index[column_index + 1]

            column_result = execute_analyzer_visits_df_csr_column_transformer(operator_context,
                                                                              self.code_reference,
                                                                              X[[column]],
                                                                              X.annotations[self],
                                                                              result[:, index_start:index_end],
                                                                              self.analyzers,
                                                                              self.code_reference_analyzer_output_map,
                                                                              description)
            annotations_for_columns = self.annotation_result_concat_workaround or []
            annotations_for_columns.append(column_result.annotations)
            self.annotation_result_concat_workaround = annotations_for_columns
        return result

    def column_transformer_visits(self, X, y):
        """
        The projections and the final concat.
        """
        # pylint: disable=invalid-name
        X_new = self.column_transformer_visits_projections(X)
        result = self.transformer.fit_transform(X_new, y)
        self.column_transformer_visits_save_child_results()
        result = self.column_transformer_visits_concat(result)
        return result

    def column_transformer_visits_concat(self, result):
        """
        Analyzer visits for the concat DAG node
        """
        # pylint: disable=too-many-locals
        result_indices = [0]
        annotations = []
        transformers_tuples = self.transformer.transformers_[:-1]
        transformers = [transformer_tuple[1] for transformer_tuple in transformers_tuples]
        for transformer in transformers:
            result_dims = transformer.output_dimensions
            annotations.extend(transformer.annotation_result_concat_workaround)
            for dim in result_dims:
                result_indices.append(result_indices[-1] + dim)
        transformer_data_with_annotations = []
        transformers_tuples = self.transformer.transformers_[:-1]
        columns_with_transformer = [(column, transformer_tuple[1]) for transformer_tuple in transformers_tuples
                                    for column in transformer_tuple[2]]
        for index, _ in enumerate(columns_with_transformer):
            data = result[:, result_indices[index]:result_indices[index + 1]]
            annotation = annotations[index]
            transformer_data_with_annotations.append((data, annotation))
        function_info = (self.module_name, "fit_transform")  # TODO: nested pipelines
        operator_context = OperatorContext(OperatorType.CONCATENATION, function_info)
        description = "concat"
        if isinstance(result, csr_matrix):
            result = execute_analyzer_visits_csr_list_csr(operator_context, self.code_reference,
                                                          transformer_data_with_annotations, result, self.analyzers,
                                                          self.code_reference_analyzer_output_map, description)
        elif isinstance(result, numpy.ndarray):
            result = execute_analyzer_visits_array_list_array(operator_context, self.code_reference,
                                                              transformer_data_with_annotations, result, self.analyzers,
                                                              self.code_reference_analyzer_output_map, description)
        else:
            assert False
        return result

    def column_transformer_visits_save_child_results(self):
        """
        Because Column transformer creates deep copies, we need to extract results here
        """
        transformers_tuples = self.transformer.transformers_[:-1]
        transformers = [transformer_tuple[1] for transformer_tuple in transformers_tuples]
        for transformer in transformers:
            self.code_reference_analyzer_output_map.update(transformer.code_reference_analyzer_output_map)

    def column_transformer_visits_projections(self, X):
        """
        Analyzer visits for the different projections
        """
        # pylint: disable=invalid-name
        transformers_tuples = self.transformer.transformers
        columns_with_transformer = [(column, transformer_tuple[1]) for transformer_tuple in transformers_tuples
                                    for column in transformer_tuple[2]]
        X_old = X.copy()
        X_new = [X]
        for column, transformer in columns_with_transformer:
            projected_df = X_old[[column]]
            function_info = (self.module_name, "fit_transform")  # TODO: nested pipelines
            operator_context = OperatorContext(OperatorType.PROJECTION, function_info)
            description = "to ['{}'] (ColumnTransformer)".format(column)
            X_new[0] = execute_analyzer_visits_df_df(operator_context, self.code_reference, X_old,
                                                     projected_df, self.analyzers,
                                                     self.code_reference_analyzer_output_map, description,
                                                     transformer, X_new[0])
        return X_new[0]

    def estimator_visits(self, X, y):
        """
        Analyzer visits for the estimator DAG node
        """
        # pylint: disable=invalid-name
        function_info = (self.module_name, "fit")  # TODO: nested pipelines
        assert y is not None
        operator_context = OperatorContext(OperatorType.ESTIMATOR, function_info)
        description = "fit"
        execute_analyzer_visits_estimator_input_nothing(operator_context, self.code_reference,
                                                        X, y, self.analyzers,
                                                        self.code_reference_analyzer_output_map, description)
        X_new = X
        y_new = y
        return X_new, y_new

    def train_data_and_labels_visits(self, X, y):
        """
        Pipeline.fit returns nothing (trained models get no edge in our DAG).
        Only need to do two scans for train data and train labels
        """
        # pylint: disable=invalid-name
        function_info = (self.module_name, "fit")  # TODO: nested pipelines
        operator_context = OperatorContext(OperatorType.TRAIN_DATA, function_info)
        X_new = execute_analyzer_visits_df_df(operator_context, self.code_reference, X, X, self.analyzers,
                                              self.code_reference_analyzer_output_map, "fit X")
        assert y is not None
        if isinstance(y, MlinspectNdarray):
            operator_context = OperatorContext(OperatorType.TRAIN_LABELS, function_info)
            y_new = execute_analyzer_visits_array_array(operator_context, self.code_reference, y, y,
                                                        self.analyzers,
                                                        self.code_reference_analyzer_output_map, "fit y")
            assert isinstance(y_new, MlinspectNdarray)
            return X_new, y_new
        elif isinstance(y, MlinspectSeries):
            operator_context = OperatorContext(OperatorType.TRAIN_LABELS, function_info)
            y_new = execute_analyzer_visits_series_series(operator_context, self.code_reference, y, y,
                                                          self.analyzers,
                                                          self.code_reference_analyzer_output_map, "fit y")
            assert isinstance(y_new, MlinspectSeries)
            return X_new, y_new
        else:
            assert False


# -------------------------------------------------------
# Functions to create the iterators for the analyzers
# -------------------------------------------------------

def iter_input_annotation_output_df_array(analyzer_index, input_data, annotations, output_data):
    """
    Create an efficient iterator for the analyzer input
    """
    # pylint: disable=too-many-locals
    # Performance tips:
    # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas

    annotation_df_view = annotations.iloc[:, analyzer_index:analyzer_index + 1]

    input_rows = get_df_row_iterator(input_data)
    annotation_rows = get_df_row_iterator(annotation_df_view)
    output_rows = get_numpy_array_row_iterator(output_data, False)

    return map(lambda input_tuple: AnalyzerInputUnaryOperator(*input_tuple),
               zip(input_rows, annotation_rows, output_rows))


def iter_input_annotation_output_df_csr(analyzer_index, input_data, annotations, output_data):
    """
    Create an efficient iterator for the analyzer input
    """
    # pylint: disable=too-many-locals
    # Performance tips:
    # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas

    annotation_df_view = annotations.iloc[:, analyzer_index:analyzer_index + 1]

    input_rows = get_df_row_iterator(input_data)
    annotation_rows = get_df_row_iterator(annotation_df_view)
    output_rows = get_csr_row_iterator(output_data)

    return map(lambda input_tuple: AnalyzerInputUnaryOperator(*input_tuple),
               zip(input_rows, annotation_rows, output_rows))


def iter_input_annotation_output_csr_list_csr(analyzer_index, transformer_data_with_annotations, output_data):
    """
    Create an efficient iterator for the analyzer input
    """
    # pylint: disable=too-many-locals
    # Performance tips:
    # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas

    input_iterators = []
    annotation_iterators = []
    for input_data, annotations in transformer_data_with_annotations:
        annotation_df_view = annotations.iloc[:, analyzer_index:analyzer_index + 1]

        input_iterators.append(get_csr_row_iterator(input_data))
        annotation_iterators.append(get_df_row_iterator(annotation_df_view))

    input_rows = map(list, zip(*input_iterators))
    annotation_rows = map(list, zip(*annotation_iterators))

    output_rows = get_csr_row_iterator(output_data)

    return map(lambda input_tuple: AnalyzerInputNAryOperator(*input_tuple),
               zip(input_rows, annotation_rows, output_rows))


def iter_input_annotation_output_array_list_array(analyzer_index, transformer_data_with_annotations, output_data):
    """
    Create an efficient iterator for the analyzer input
    """
    # pylint: disable=too-many-locals
    # Performance tips:
    # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas

    input_iterators = []
    annotation_iterators = []
    for input_data, annotations in transformer_data_with_annotations:
        annotation_df_view = annotations.iloc[:, analyzer_index:analyzer_index + 1]

        input_iterators.append(get_numpy_array_row_iterator(input_data))
        annotation_iterators.append(get_df_row_iterator(annotation_df_view))

    input_rows = map(list, zip(*input_iterators))
    annotation_rows = map(list, zip(*annotation_iterators))

    output_rows = get_numpy_array_row_iterator(output_data, False)

    return map(lambda input_tuple: AnalyzerInputNAryOperator(*input_tuple),
               zip(input_rows, annotation_rows, output_rows))


def iter_input_annotation_output_estimator_nothing(analyzer_index, data, target):
    """
    Create an efficient iterator for the analyzer input
    """
    # pylint: disable=too-many-locals
    # Performance tips:
    # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas

    input_iterators = []
    annotation_iterators = []

    data_annotation_df_view = data.annotations.iloc[:, analyzer_index:analyzer_index + 1]
    if isinstance(data, MlinspectCsrMatrix):
        input_iterators.append(get_csr_row_iterator(data))
    elif isinstance(data, MlinspectNdarray):
        input_iterators.append(get_numpy_array_row_iterator(data, False))
    else:
        assert False
    annotation_iterators.append(get_df_row_iterator(data_annotation_df_view))

    target_annotation_df_view = target.annotations.iloc[:, analyzer_index:analyzer_index + 1]
    if isinstance(target, MlinspectNdarray):
        input_iterators.append(get_numpy_array_row_iterator(target))
    elif isinstance(target, MlinspectSeries):
        input_iterators.append(get_series_row_iterator(target))
    else:
        assert False
    annotation_iterators.append(get_df_row_iterator(target_annotation_df_view))

    input_rows = map(list, zip(*input_iterators))
    annotation_rows = map(list, zip(*annotation_iterators))

    return map(lambda input_tuple: AnalyzerInputSinkOperator(*input_tuple),
               zip(input_rows, annotation_rows))


def iter_input_annotation_output_df_df(analyzer_index, input_df, annotation_df, output_df):
    """
    Create an efficient iterator for the analyzer input
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
    Create an efficient iterator for the analyzer input
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


def iter_input_annotation_output_series_series(analyzer_index, input_series, annotation_df, output_series):
    """
    Create an efficient iterator for the analyzer input
    """
    # pylint: disable=too-many-locals
    # Performance tips:
    # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
    assert isinstance(input_series, MlinspectSeries)
    assert isinstance(output_series, MlinspectSeries)

    annotation_df_view = annotation_df.iloc[:, analyzer_index:analyzer_index + 1]

    input_rows = get_series_row_iterator(input_series)
    annotation_rows = get_df_row_iterator(annotation_df_view)
    output_rows = get_series_row_iterator(output_series)

    return map(lambda input_tuple: AnalyzerInputUnaryOperator(*input_tuple),
               zip(input_rows, annotation_rows, output_rows))

# -------------------------------------------------------
# Execute analyzers functions
# -------------------------------------------------------


def execute_analyzer_visits_csr_list_csr(operator_context, code_reference, transformer_data_with_annotations,
                                         output_data, analyzers,
                                         code_reference_analyzer_output_map, func_name):
    """Execute analyzers"""
    # pylint: disable=too-many-arguments
    annotation_iterators = []
    for analyzer in analyzers:
        analyzer_index = analyzers.index(analyzer)
        iterator_for_analyzer = iter_input_annotation_output_csr_list_csr(analyzer_index,
                                                                          transformer_data_with_annotations,
                                                                          output_data)
        annotations_iterator = analyzer.visit_operator(operator_context, iterator_for_analyzer)
        annotation_iterators.append(annotations_iterator)
    return_value = store_analyzer_outputs_csr(annotation_iterators, code_reference, output_data, analyzers,
                                              code_reference_analyzer_output_map, func_name)
    assert isinstance(return_value, MlinspectCsrMatrix)
    return return_value


def execute_analyzer_visits_array_list_array(operator_context, code_reference, transformer_data_with_annotations,
                                             output_data, analyzers,
                                             code_reference_analyzer_output_map, func_name):
    """Execute analyzers"""
    # pylint: disable=too-many-arguments
    annotation_iterators = []
    for analyzer in analyzers:
        analyzer_index = analyzers.index(analyzer)
        iterator_for_analyzer = iter_input_annotation_output_array_list_array(analyzer_index,
                                                                              transformer_data_with_annotations,
                                                                              output_data)
        annotations_iterator = analyzer.visit_operator(operator_context, iterator_for_analyzer)
        annotation_iterators.append(annotations_iterator)
    return_value = store_analyzer_outputs_array(annotation_iterators, code_reference, output_data, analyzers,
                                                code_reference_analyzer_output_map, func_name)
    assert isinstance(return_value, MlinspectNdarray)
    return return_value


def execute_analyzer_visits_estimator_input_nothing(operator_context, code_reference, data, target,
                                                    analyzers, code_reference_analyzer_output_map, func_name):
    """Execute analyzers"""
    # pylint: disable=too-many-arguments
    assert isinstance(data, MlinspectCsrMatrix) or isinstance(data, MlinspectNdarray)
    assert isinstance(target, MlinspectNdarray) or isinstance(target, MlinspectSeries)
    annotation_iterators = []
    for analyzer in analyzers:
        analyzer_index = analyzers.index(analyzer)
        iterator_for_analyzer = iter_input_annotation_output_estimator_nothing(analyzer_index, data, target)
        annotations_iterator = analyzer.visit_operator(operator_context, iterator_for_analyzer)
        annotation_iterators.append(annotations_iterator)
    store_analyzer_outputs_estimator(annotation_iterators, code_reference, analyzers,
                                     code_reference_analyzer_output_map, func_name)


def execute_analyzer_visits_df_df(operator_context, code_reference, input_data, output_data, analyzers,
                                  code_reference_analyzer_output_map, func_name, transformer=None,
                                  full_return_value=None):
    """Execute analyzers"""
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
                                             code_reference_analyzer_output_map, func_name, transformer,
                                             full_return_value)
    assert isinstance(return_value, MlinspectDataFrame)
    return return_value


def execute_analyzer_visits_df_array_column_transformer(operator_context, code_reference,
                                                        input_data, annotations, output_data, analyzers,
                                                        code_reference_analyzer_output_map,
                                                        func_name):
    """Execute analyzers"""
    # pylint: disable=too-many-arguments
    assert isinstance(input_data, MlinspectDataFrame)
    annotation_iterators = []
    for analyzer in analyzers:
        analyzer_index = analyzers.index(analyzer)
        iterator_for_analyzer = iter_input_annotation_output_df_array(analyzer_index,
                                                                      input_data,
                                                                      annotations,
                                                                      output_data)
        annotations_iterator = analyzer.visit_operator(operator_context, iterator_for_analyzer)
        annotation_iterators.append(annotations_iterator)
    return_value = store_analyzer_outputs_array(annotation_iterators, code_reference, output_data, analyzers,
                                                code_reference_analyzer_output_map, func_name)
    assert isinstance(return_value, MlinspectNdarray)
    return return_value


def execute_analyzer_visits_df_csr_column_transformer(operator_context, code_reference,
                                                      input_data, annotations, output_data, analyzers,
                                                      code_reference_analyzer_output_map,
                                                      func_name):
    """Execute analyzers"""
    # pylint: disable=too-many-arguments
    assert isinstance(input_data, MlinspectDataFrame)
    annotation_iterators = []
    for analyzer in analyzers:
        analyzer_index = analyzers.index(analyzer)
        iterator_for_analyzer = iter_input_annotation_output_df_csr(analyzer_index,
                                                                    input_data,
                                                                    annotations,
                                                                    output_data)
        annotations_iterator = analyzer.visit_operator(operator_context, iterator_for_analyzer)
        annotation_iterators.append(annotations_iterator)
    return_value = store_analyzer_outputs_csr(annotation_iterators, code_reference, output_data, analyzers,
                                              code_reference_analyzer_output_map, func_name)
    assert isinstance(return_value, MlinspectCsrMatrix)
    return return_value


def execute_analyzer_visits_array_array(operator_context, code_reference, input_data, output_data,
                                        analyzers, code_reference_analyzer_output_map, func_name):
    """Execute analyzers"""
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


def execute_analyzer_visits_series_series(operator_context, code_reference, input_data, output_data,
                                          analyzers, code_reference_analyzer_output_map, func_name):
    """Execute analyzers"""
    # pylint: disable=too-many-arguments
    assert isinstance(input_data, MlinspectSeries)
    annotation_iterators = []
    for analyzer in analyzers:
        analyzer_index = analyzers.index(analyzer)
        iterator_for_analyzer = iter_input_annotation_output_series_series(analyzer_index,
                                                                           input_data,
                                                                           input_data.annotations,
                                                                           output_data)
        annotations_iterator = analyzer.visit_operator(operator_context, iterator_for_analyzer)
        annotation_iterators.append(annotations_iterator)
    return_value = store_analyzer_outputs_series(annotation_iterators, code_reference, output_data, analyzers,
                                                 code_reference_analyzer_output_map, func_name)
    assert isinstance(return_value, MlinspectSeries)
    return return_value

# -------------------------------------------------------
# Store analyzer results functions
# -------------------------------------------------------


def store_analyzer_outputs_df(annotation_iterators, code_reference, return_value, analyzers,
                              code_reference_analyzer_output_map, func_name, transformer=None, full_return_value=None):
    """
    Stores the analyzer annotations for the rows in the dataframe and the
    analyzer annotations for the DAG operators in a map
    """
    # pylint: disable=too-many-arguments, too-many-locals
    annotations_df = build_annotation_df_from_iters(analyzers, annotation_iterators)
    analyzer_outputs = {}
    for analyzer in analyzers:
        analyzer_output = analyzer.get_operator_annotation_after_visit()
        analyzer_outputs[analyzer] = analyzer_output

    stored_analyzer_results = code_reference_analyzer_output_map.get(code_reference, {})
    stored_analyzer_results[func_name] = analyzer_outputs
    code_reference_analyzer_output_map[code_reference] = stored_analyzer_results

    # If the transformer is a column transformer, we have multiple annotations we need to pass to different transformers
    # If we do not want to override internal column transformer functions, we have to work around these black
    # box functions and pass the annotations using a different mechanism
    if transformer is None:
        assert full_return_value is None
        new_return_value = MlinspectDataFrame(return_value)
        new_return_value.annotations = annotations_df
    else:
        if not isinstance(full_return_value, MlinspectDataFrame):
            new_return_value = MlinspectDataFrame(full_return_value)
        else:
            new_return_value = full_return_value
        if not hasattr(new_return_value, "annotations") or not isinstance(new_return_value.annotations, dict):
            new_return_value.annotations = dict()
        new_return_value.annotations[transformer] = annotations_df
    assert isinstance(new_return_value, MlinspectDataFrame)
    return new_return_value


def store_analyzer_outputs_array(annotation_iterators, code_reference, return_value, analyzers,
                                 code_reference_analyzer_output_map, func_name):
    """
    Stores the analyzer annotations for the rows in the dataframe and the
    analyzer annotations for the DAG operators in a map
    """
    # pylint: disable=too-many-arguments
    annotations_df = build_annotation_df_from_iters(analyzers, annotation_iterators)
    analyzer_outputs = {}
    for analyzer in analyzers:
        analyzer_outputs[analyzer] = analyzer.get_operator_annotation_after_visit()

    stored_analyzer_results = code_reference_analyzer_output_map.get(code_reference, {})
    stored_analyzer_results[func_name] = analyzer_outputs
    code_reference_analyzer_output_map[code_reference] = stored_analyzer_results
    return_value = MlinspectNdarray(return_value)
    return_value.annotations = annotations_df
    assert isinstance(return_value, MlinspectNdarray)
    return return_value


def store_analyzer_outputs_series(annotation_iterators, code_reference, return_value, analyzers,
                                  code_reference_analyzer_output_map, func_name):
    """
    Stores the analyzer annotations for the rows in the dataframe and the
    analyzer annotations for the DAG operators in a map
    """
    # pylint: disable=too-many-arguments
    annotations_df = build_annotation_df_from_iters(analyzers, annotation_iterators)
    analyzer_outputs = {}
    for analyzer in analyzers:
        analyzer_outputs[analyzer] = analyzer.get_operator_annotation_after_visit()

    stored_analyzer_results = code_reference_analyzer_output_map.get(code_reference, {})
    stored_analyzer_results[func_name] = analyzer_outputs
    code_reference_analyzer_output_map[code_reference] = stored_analyzer_results
    return_value = MlinspectSeries(return_value)
    return_value.annotations = annotations_df
    assert isinstance(return_value, MlinspectSeries)
    return return_value


def store_analyzer_outputs_csr(annotation_iterators, code_reference, return_value, analyzers,
                               code_reference_analyzer_output_map, func_name):
    """
    Stores the analyzer annotations for the rows in the dataframe and the
    analyzer annotations for the DAG operators in a map
    """
    # pylint: disable=too-many-arguments
    annotations_df = build_annotation_df_from_iters(analyzers, annotation_iterators)
    analyzer_outputs = {}
    for analyzer in analyzers:
        analyzer_output = analyzer.get_operator_annotation_after_visit()
        analyzer_outputs[analyzer] = analyzer_output

    stored_analyzer_results = code_reference_analyzer_output_map.get(code_reference, {})
    stored_analyzer_results[func_name] = analyzer_outputs
    code_reference_analyzer_output_map[code_reference] = stored_analyzer_results
    return_value = MlinspectCsrMatrix(return_value)
    return_value.annotations = annotations_df
    assert isinstance(return_value, MlinspectCsrMatrix)
    return return_value


def store_analyzer_outputs_estimator(annotation_iterators, code_reference, analyzers,
                                     code_reference_analyzer_output_map, func_name):
    """
    Stores the analyzer annotations for the rows in the dataframe and the
    analyzer annotations for the DAG operators in a map
    """
    # pylint: disable=too-many-arguments
    build_annotation_df_from_iters(analyzers, annotation_iterators)

    analyzer_outputs = {}
    for analyzer in analyzers:
        analyzer_output = analyzer.get_operator_annotation_after_visit()
        analyzer_outputs[analyzer] = analyzer_output

    stored_analyzer_results = code_reference_analyzer_output_map.get(code_reference, {})
    stored_analyzer_results[func_name] = analyzer_outputs
    code_reference_analyzer_output_map[code_reference] = stored_analyzer_results
