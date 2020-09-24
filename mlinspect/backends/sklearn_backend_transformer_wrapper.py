"""
A wrapper for sklearn transformers to capture method calls we do not see otherwise because of the pipeline
definition style
"""
import inspect
import uuid
from enum import Enum

from sklearn.base import BaseEstimator

from .backend_utils import get_df_row_iterator, \
    build_annotation_df_from_iters, get_iterator_for_type, create_wrapper_with_annotations
from .pandas_backend_frame_wrapper import MlinspectDataFrame, MlinspectSeries
from .sklearn_backend_csr_matrx_wrapper import MlinspectCsrMatrix
from .sklearn_backend_ndarray_wrapper import MlinspectNdarray
from ..inspections.inspection_input import OperatorContext, InspectionInputUnaryOperator, \
    InspectionInputNAryOperator, InspectionInputSinkOperator
from ..instrumentation.dag_node import CodeReference, OperatorType


class MlinspectEstimatorTransformer(BaseEstimator):
    """
    A wrapper for sklearn transformers to capture method calls we do not see otherwise because of the pipeline
    definition style
    See: https://scikit-learn.org/stable/developers/develop.html
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, transformer, code_reference: CodeReference, inspections, code_ref_inspection_output_map,
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
        self.inspections = inspections
        self.code_ref_inspection_output_map = code_ref_inspection_output_map
        self.output_dimensions = output_dimensions
        self.annotation_result_concat_workaround = None
        self.parent_transformer = None  # needed for nested pipelines
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
            self.transformer = self.transformer.fit(X_new, y_new)
        elif self.call_function_info in {('sklearn.tree._classes', 'DecisionTreeClassifier'),
                                         ('tensorflow.python.keras.wrappers.scikit_learn', 'KerasClassifier')}:
            X_new, y_new = self.estimator_visits(X, y)
            self.transformer.fit(X_new, y_new)
        else:
            assert False

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
        elif self.call_function_info == ('demo.healthcare.demo_utils', 'MyW2VTransformer'):
            result = self.w2v_visits(X, y)
        elif self.call_function_info == ('sklearn.impute._base', 'SimpleImputer'):
            result = self.simple_imputer_visits(X, y)
        elif self.call_function_info == ('sklearn.pipeline', 'Pipeline'):
            self.transformer.steps[0][1].parent_transformer = self
            result = self.transformer.fit_transform(X, y)
            last_step_transformer = self.transformer.steps[-1][1]
            self.annotation_result_concat_workaround = last_step_transformer.annotation_result_concat_workaround
            self.output_dimensions = last_step_transformer.output_dimensions

            transformers_tuples = self.transformer.steps
            transformers = [transformer_tuple[1] for transformer_tuple in transformers_tuples]
            for transformer in transformers:
                self.code_ref_inspection_output_map.update(transformer.code_ref_inspection_output_map)
        else:
            result = self.transformer.fit_transform(X, y)

        return result

    def standard_scaler_visits(self, X, y):
        """
        Inspection visits for the StandardScaler Transformer
        """
        # pylint: disable=invalid-name
        assert isinstance(X.annotations, dict) and self in X.annotations
        result = self.transformer.fit_transform(X, y)
        self.output_dimensions = [1 for _ in range(result.shape[1])]
        for column_index, column in enumerate(X.columns):
            function_info = (self.module_name, "fit_transform")  # TODO: nested pipelines
            operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
            description = "Numerical Encoder (StandardScaler), Column: '{}'".format(column)
            column_result = execute_inspection_visits_unary_op(operator_context,
                                                               self.code_reference,
                                                               X[[column]],
                                                               X.annotations[self],
                                                               result[:, column_index],
                                                               self.inspections,
                                                               self.code_ref_inspection_output_map,
                                                               description)
            annotations_for_columns = self.annotation_result_concat_workaround or []
            annotations_for_columns.append(column_result.annotations)
            self.annotation_result_concat_workaround = annotations_for_columns
        return result

    def simple_imputer_visits(self, X, y):
        """
        Inspection visits for the StandardScaler Transformer
        """
        # pylint: disable=invalid-name
        assert isinstance(X.annotations, dict) and self.parent_transformer in X.annotations
        result = self.transformer.fit_transform(X, y)
        self.output_dimensions = [1 for _ in range(result.shape[1])]
        for column_index, column in enumerate(X.columns):
            function_info = (self.module_name, "fit_transform")  # TODO: nested pipelines
            operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
            description = "Imputer (SimpleImputer), Column: '{}'".format(column)
            column_result = execute_inspection_visits_unary_op(operator_context,
                                                               self.code_reference,
                                                               X[[column]],
                                                               X.annotations[
                                                                   self.parent_transformer],
                                                               result[:, column_index],
                                                               self.inspections,
                                                               self.code_ref_inspection_output_map,
                                                               description)
            annotations_for_columns = self.annotation_result_concat_workaround or []
            annotations_for_columns.append((column, column_result.annotations))
            self.annotation_result_concat_workaround = annotations_for_columns
        result = MlinspectNdarray(result)
        result.annotations = self.annotation_result_concat_workaround
        return result

    def w2v_visits(self, X, y):
        """
        Inspection visits for the StandardScaler Transformer
        """
        # pylint: disable=invalid-name
        assert isinstance(X.annotations, dict) and self in X.annotations
        result = self.transformer.fit_transform(X, y)
        self.output_dimensions = [result.shape[1]]
        for column in X.columns:
            function_info = (self.module_name, "fit_transform")  # TODO: could also be used for multiple columns at once
            operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
            description = "Word2Vec, Column: '{}'".format(column)
            column_result = execute_inspection_visits_unary_op(operator_context,
                                                               self.code_reference,
                                                               X[[column]],
                                                               X.annotations[self],
                                                               result[:, :],
                                                               self.inspections,
                                                               self.code_ref_inspection_output_map,
                                                               description)
            annotations_for_columns = self.annotation_result_concat_workaround or []
            annotations_for_columns.append(column_result.annotations)
            self.annotation_result_concat_workaround = annotations_for_columns
        return result

    def one_hot_encoder_visits(self, X, y):
        """
        Inspection visits for the OneHotEncoder Transformer
        """
        # pylint: disable=invalid-name, too-many-locals
        transformer_name = "Categorical Encoder (OneHotEncoder)"
        result = self.transformer.fit_transform(X, y)
        self.output_dimensions = [len(one_hot_categories) for one_hot_categories in
                                  self.transformer.categories_]

        output_dimension_index = [0]
        for dimension in self.output_dimensions:
            output_dimension_index.append(output_dimension_index[-1] + dimension)

        function_info = (self.module_name, "fit_transform")
        operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)

        for column_index in range(X.shape[1]):
            if isinstance(X.annotations, dict):  # A dict is used for ColumnTransformer projections
                assert isinstance(X, MlinspectDataFrame) and self in X.annotations
                column_name = X.columns[column_index]
                annotations = X.annotations[self]
                input_data = X.iloc[:, column_index]
            elif isinstance(X.annotations, list):  # List because transformer impls process multiple columns at once
                assert isinstance(X, MlinspectNdarray)
                column_name, annotations = X.annotations[column_index]
                input_data = X[:, column_index]
            else:
                assert False

            description = "{}, Column: '{}'".format(transformer_name, column_name)
            index_start = output_dimension_index[column_index]
            index_end = output_dimension_index[column_index + 1]
            column_result = execute_inspection_visits_unary_op(operator_context,
                                                               self.code_reference,
                                                               input_data,
                                                               annotations,
                                                               result[:, index_start:index_end],
                                                               self.inspections,
                                                               self.code_ref_inspection_output_map,
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
        Inspection visits for the concat DAG node
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
        result = execute_inspection_visits_nary_op(operator_context, self.code_reference,
                                                   transformer_data_with_annotations, result, self.inspections,
                                                   self.code_ref_inspection_output_map, description)
        return result

    def column_transformer_visits_save_child_results(self):
        """
        Because Column transformer creates deep copies, we need to extract results here
        """
        transformers_tuples = self.transformer.transformers_[:-1]
        transformers = [transformer_tuple[1] for transformer_tuple in transformers_tuples]
        for transformer in transformers:
            self.code_ref_inspection_output_map.update(transformer.code_ref_inspection_output_map)

    def column_transformer_visits_projections(self, X):
        """
        Inspection visits for the different projections
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
            X_new[0] = execute_inspection_visits_unary_op(operator_context, self.code_reference, X_old,
                                                          X_old.annotations, projected_df, self.inspections,
                                                          self.code_ref_inspection_output_map, description,
                                                          True, transformer, X_new[0])
        return X_new[0]

    def estimator_visits(self, X, y):
        """
        Inspection visits for the estimator DAG node
        """
        # pylint: disable=invalid-name
        function_info = (self.module_name, "fit")  # TODO: nested pipelines
        assert y is not None
        operator_context = OperatorContext(OperatorType.ESTIMATOR, function_info)
        description = "fit"
        execute_inspection_visits_sink_op(operator_context, self.code_reference,
                                          X, y, self.inspections,
                                          self.code_ref_inspection_output_map, description)
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
        X_new = execute_inspection_visits_unary_op(operator_context, self.code_reference, X, X.annotations, X,
                                                   self.inspections, self.code_ref_inspection_output_map, "fit X")
        assert y is not None
        operator_context = OperatorContext(OperatorType.TRAIN_LABELS, function_info)
        y_new = execute_inspection_visits_unary_op(operator_context, self.code_reference, y,
                                                   y.annotations, y, self.inspections,
                                                   self.code_ref_inspection_output_map, "fit y")
        result = X_new, y_new
        return result

    def score(self, X, y):
        """
        Forward some score call of an estimator
        """
        # pylint: disable=invalid-name
        # TODO: Probably split the transformer_estimator wrapper into two for transforemrs and estimators
        return self.transformer.score(X, y)


# -------------------------------------------------------
# Functions to create the iterators for the inspections
# -------------------------------------------------------

def iter_input_annotation_output_nary_op(inspection_index, transformer_data_with_annotations, output_data):
    """
    Create an efficient iterator for the inspection input
    """
    # pylint: disable=too-many-locals
    # Performance tips:
    # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas

    input_iterators = []
    annotation_iterators = []
    for input_data, annotations in transformer_data_with_annotations:
        annotation_df_view = annotations.iloc[:, inspection_index:inspection_index + 1]
        input_iterators.append(get_iterator_for_type(input_data, True))
        annotation_iterators.append(get_df_row_iterator(annotation_df_view))

    input_rows = map(list, zip(*input_iterators))
    annotation_rows = map(list, zip(*annotation_iterators))
    output_rows = get_iterator_for_type(output_data, False)

    return map(lambda input_tuple: InspectionInputNAryOperator(*input_tuple),
               zip(input_rows, annotation_rows, output_rows))


def iter_input_annotation_output_sink_op(inspection_index, data, target):
    """
    Create an efficient iterator for the inspection input
    """
    # pylint: disable=too-many-locals
    # Performance tips:
    # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas

    input_iterators = []
    annotation_iterators = []

    data_annotation_df_view = data.annotations.iloc[:, inspection_index:inspection_index + 1]
    input_iterators.append(get_iterator_for_type(data, False))
    annotation_iterators.append(get_df_row_iterator(data_annotation_df_view))

    target_annotation_df_view = target.annotations.iloc[:, inspection_index:inspection_index + 1]
    input_iterators.append(get_iterator_for_type(target, True))
    annotation_iterators.append(get_iterator_for_type(target_annotation_df_view))

    input_rows = map(list, zip(*input_iterators))
    annotation_rows = map(list, zip(*annotation_iterators))

    return map(lambda input_tuple: InspectionInputSinkOperator(*input_tuple),
               zip(input_rows, annotation_rows))


def iter_input_annotation_output_unary_op(inspection_index, input_data, input_annotations, output):
    """
    Create an efficient iterator for the inspection input
    """
    # pylint: disable=too-many-locals
    # Performance tips:
    # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
    input_rows = get_iterator_for_type(input_data, True)

    annotation_df_view = input_annotations.iloc[:, inspection_index:inspection_index + 1]
    annotation_rows = get_df_row_iterator(annotation_df_view)

    output_rows = get_iterator_for_type(output, False)

    return map(lambda input_tuple: InspectionInputUnaryOperator(*input_tuple),
               zip(input_rows, annotation_rows, output_rows))


# -------------------------------------------------------
# Execute inspections functions
# -------------------------------------------------------

def execute_inspection_visits_nary_op(operator_context, code_reference, transformer_data_with_annotations,
                                      output_data, inspections,
                                      code_ref_inspection_output_map, func_name):
    """Execute inspections"""
    # pylint: disable=too-many-arguments
    annotation_iterators = []
    for inspection in inspections:
        inspection_index = inspections.index(inspection)
        iterator_for_inspection = iter_input_annotation_output_nary_op(inspection_index,
                                                                       transformer_data_with_annotations,
                                                                       output_data)
        annotations_iterator = inspection.visit_operator(operator_context, iterator_for_inspection)
        annotation_iterators.append(annotations_iterator)
    return_value = store_inspection_outputs(annotation_iterators, code_reference, output_data, inspections,
                                            code_ref_inspection_output_map, func_name, StorageType.NORMAL)
    return return_value


def execute_inspection_visits_sink_op(operator_context, code_reference, data, target,
                                      inspections, code_reference_inspection_output_map, func_name):
    """Execute inspections"""
    # pylint: disable=too-many-arguments
    assert isinstance(data, (MlinspectCsrMatrix, MlinspectNdarray))
    assert isinstance(target, (MlinspectNdarray, MlinspectSeries))
    annotation_iterators = []
    for inspection in inspections:
        inspection_index = inspections.index(inspection)
        iterator_for_inspection = iter_input_annotation_output_sink_op(inspection_index, data, target)
        annotations_iterator = inspection.visit_operator(operator_context, iterator_for_inspection)
        annotation_iterators.append(annotations_iterator)
    store_inspection_outputs(annotation_iterators, code_reference, None, inspections,
                             code_reference_inspection_output_map, func_name, StorageType.ESTIMATOR)


def execute_inspection_visits_unary_op(operator_context, code_reference, input_data, input_annotations, output_data,
                                       inspections, code_reference_inspection_output_map, func_name,
                                       column_transformer_projection=False, transformer=None, full_return_value=None):
    """Execute inspections"""
    # pylint: disable=too-many-arguments, too-many-locals
    annotation_iterators = []
    for inspection in inspections:
        inspection_index = inspections.index(inspection)
        iterator_for_inspection = iter_input_annotation_output_unary_op(inspection_index,
                                                                        input_data,
                                                                        input_annotations,
                                                                        output_data)
        annotations_iterator = inspection.visit_operator(operator_context, iterator_for_inspection)
        annotation_iterators.append(annotations_iterator)
    if not column_transformer_projection:
        return_value = store_inspection_outputs(annotation_iterators, code_reference, output_data, inspections,
                                                code_reference_inspection_output_map, func_name, StorageType.NORMAL)
    else:
        return_value = store_inspection_outputs(annotation_iterators, code_reference, output_data, inspections,
                                                code_reference_inspection_output_map, func_name,
                                                StorageType.COLUMN_TRANSFORMER, transformer, full_return_value)
    return return_value


# -------------------------------------------------------
# Store inspection results functions
# -------------------------------------------------------

class StorageType(Enum):
    """
    The different operator types in our DAG
    """
    ESTIMATOR = 1
    COLUMN_TRANSFORMER = 2
    NORMAL = 3


def store_inspection_outputs(annotation_iterators, code_reference, return_value, inspections,
                             code_reference_inspection_output_map, func_name, storage_type,
                             transformer=None, full_return_value=None):
    """
    Stores the inspection annotations for the rows in the dataframe and the
    inspection annotations for the DAG operators in a map
    """
    # pylint: disable=too-many-arguments, too-many-locals, too-many-branches
    annotations_df = build_annotation_df_from_iters(inspections, annotation_iterators)
    inspection_outputs = {}
    for inspection in inspections:
        inspection_outputs[inspection] = inspection.get_operator_annotation_after_visit()

    stored_inspection_results = code_reference_inspection_output_map.get(code_reference, {})
    stored_inspection_results[func_name] = inspection_outputs
    code_reference_inspection_output_map[code_reference] = stored_inspection_results

    if storage_type == StorageType.COLUMN_TRANSFORMER:
        # If the transformer is a column transformer, we have multiple annotations we need to pass to different
        # transformers.  If we do not want to override internal column transformer functions, we have to work around
        # these black box functions and pass the annotations using a different mechanism
        if not isinstance(full_return_value, MlinspectDataFrame):
            new_return_value = MlinspectDataFrame(full_return_value)
        else:
            new_return_value = full_return_value
        if not hasattr(new_return_value, "annotations") or not isinstance(new_return_value.annotations, dict):
            new_return_value.annotations = dict()
        new_return_value.annotations[transformer] = annotations_df
        assert isinstance(new_return_value, MlinspectDataFrame)
    elif storage_type == StorageType.ESTIMATOR:
        new_return_value = None
    else:
        new_return_value = create_wrapper_with_annotations(annotations_df, return_value)
    return new_return_value
