"""
A wrapper for sklearn transformers to capture method calls we do not see otherwise because of the pipeline
definition style
"""
import inspect

import numpy
from pandas import DataFrame
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator

from ._backend_utils import create_wrapper_with_annotations, \
    build_annotation_list_from_iters
from ._iter_creation import iter_input_annotation_output_sink_op, iter_input_annotation_output_nary_op, \
    iter_input_annotation_output_map
from ._pandas_backend_frame_wrapper import MlinspectDataFrame, MlinspectSeries
from ._sklearn_backend_csr_matrx_wrapper import MlinspectCsrMatrix
from ._sklearn_backend_ndarray_wrapper import MlinspectNdarray
from ..inspections._inspection_input import OperatorContext
from ..instrumentation._dag_node import CodeReference, OperatorType

transformer_names = {
        ('sklearn.preprocessing._encoders', 'OneHotEncoder'): "Categorical Encoder (OneHotEncoder)",
        ('sklearn.preprocessing._data', 'StandardScaler'): "Numerical Encoder (StandardScaler)",
        ('sklearn.impute._base', 'SimpleImputer'): "Imputer (SimpleImputer)",
        ('sklearn.tree._classes', 'DecisionTreeClassifier'): "Decision Tree",
        ('sklearn.linear_model._logistic', 'LogisticRegression'): "Logistic Regression",
        ('sklearn.preprocessing._discretization', 'KBinsDiscretizer'): "Numerical Encoder (KBinsDiscretizer)",
        # TODO: We  can remove this later by checking if subclass of transformer/estimator
        ('example_pipelines.healthcare.healthcare_utils', 'MyW2VTransformer'): "Word2Vec",
        ('tensorflow.python.keras.wrappers.scikit_learn', 'KerasClassifier'): "Neural Network"
    }


class MlinspectEstimatorTransformer(BaseEstimator):
    """
    A wrapper for sklearn transformers to capture method calls we do not see otherwise because of the pipeline
    definition style
    See: https://scikit-learn.org/stable/developers/develop.html
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, transformer, code_reference: CodeReference, inspections, code_ref_inspection_output_map,
                 output_dimensions=None, annotation_result_project_workaround=None):
        # pylint: disable=too-many-arguments
        # None arguments are not passed directly when we create them. Still needed though because the
        # Column transformer clones child transformers and does not pass parameters otherwise
        module = inspect.getmodule(transformer)

        self.transformer = transformer
        self.module_name = module.__name__
        self.call_function_info = (module.__name__, transformer.__class__.__name__)
        self.code_reference = code_reference
        self.inspections = inspections
        self.code_ref_inspection_output_map = code_ref_inspection_output_map
        self.output_dimensions = output_dimensions
        self.annotation_result_concat_workaround = None
        self.annotation_result_project_workaround = annotation_result_project_workaround

    def fit(self, X, y=None) -> 'MlinspectEstimatorTransformer':
        """
        Override fit
        """
        # pylint: disable=invalid-name
        if self.call_function_info == ('sklearn.pipeline', 'Pipeline'):
            X_annotated, y_annotated = self.train_data_and_labels_visits(X, y)
            self.transformer = self.transformer.fit(X_annotated, y_annotated)
        elif self.call_function_info in {('sklearn.tree._classes', 'DecisionTreeClassifier'),
                                         ('tensorflow.python.keras.wrappers.scikit_learn', 'KerasClassifier'),
                                         ('sklearn.linear_model._logistic', 'LogisticRegression')}:
            self.estimator_visits(X, y)
            self.transformer.fit(X, y)
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
        elif self.call_function_info == ('sklearn.pipeline', 'Pipeline'):
            result = self.pipeline_visit(X, y)
        elif self.call_function_info in {('sklearn.preprocessing._data', 'StandardScaler'),
                                         ('sklearn.impute._base', 'SimpleImputer'),
                                         ('sklearn.preprocessing._discretization', 'KBinsDiscretizer')}:
            result = self.transformer.fit_transform(X, y)
            self.output_dimensions = [1 for _ in range(result.shape[1])]
            result = self.normal_transformer_visit(X, y, result)
        elif self.call_function_info == ('example_pipelines.healthcare.healthcare_utils', 'MyW2VTransformer'):
            result = self.transformer.fit_transform(X, y)
            self.output_dimensions = [result.shape[1]]
            result = self.normal_transformer_visit(X, y, result)
        elif self.call_function_info == ('sklearn.preprocessing._encoders', 'OneHotEncoder'):
            result = self.transformer.fit_transform(X, y)
            self.output_dimensions = [len(one_hot_categories) for one_hot_categories in
                                      self.transformer.categories_]
            result = self.normal_transformer_visit(X, y, result)
        else:
            result = self.transformer.fit_transform(X, y)

        return result

    def normal_transformer_visit(self, X, y, result):
        """
        Inspection visits for the OneHotEncoder Transformer
        """
        # pylint: disable=invalid-name, too-many-locals, too-many-arguments, unused-argument
        transformer_name = transformer_names[self.call_function_info]

        output_dimension_index = [0]
        assert self.output_dimensions
        for dimension in self.output_dimensions:
            output_dimension_index.append(output_dimension_index[-1] + dimension)

        function_info = (self.module_name, "fit_transform")
        operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)

        for column_index in range(X.shape[1]):
            if self.annotation_result_project_workaround is not None:
                assert isinstance(X, MlinspectDataFrame)
                column_name = X.columns[column_index]
                annotations = self.annotation_result_project_workaround[column_index]
                input_data = X[[column_name]]
            elif isinstance(X.annotations, list):  # List because transformer impls process multiple columns at once
                assert isinstance(X, MlinspectNdarray)
                column_name, annotations = X.annotations[column_index]
                input_data = X[:, column_index]
            elif isinstance(X.annotations, DataFrame) and len(X.columns) == 1:
                # Currently, this is only used when benchmarking single transformers
                # TODO: Full support for calling fit_transform etc. on encoders directly instead of using
                #  column transformers
                column_name = X.columns[0]
                annotations = X.annotations
                input_data = X
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
                                                               description,
                                                               [column_name])
            annotations_for_columns = self.annotation_result_concat_workaround or []
            annotations_for_columns.append((column_name, column_result.annotations))
            self.annotation_result_concat_workaround = annotations_for_columns
        if isinstance(result, numpy.ndarray):
            result = MlinspectNdarray(result)
            result.annotations = self.annotation_result_concat_workaround
        elif isinstance(result, csr_matrix):
            result = MlinspectCsrMatrix(result)
            result.annotations = self.annotation_result_concat_workaround
        else:
            assert False
        return result

    def pipeline_visit(self, X, y):
        """
        Inspection visits for a Pipeline within a Pipeline
        """
        # pylint: disable=invalid-name
        if self.annotation_result_project_workaround is not None:
            first_step_transformer = self.transformer.steps[0][1]
            first_step_transformer.annotation_result_project_workaround = self.annotation_result_project_workaround
        result = self.transformer.fit_transform(X, y)
        last_step_transformer = self.transformer.steps[-1][1]
        self.annotation_result_concat_workaround = last_step_transformer.annotation_result_concat_workaround
        self.output_dimensions = last_step_transformer.output_dimensions
        transformers_tuples = self.transformer.steps
        transformers = [transformer_tuple[1] for transformer_tuple in transformers_tuples]
        for transformer in transformers:
            self.code_ref_inspection_output_map.update(transformer.code_ref_inspection_output_map)
        return result

    def column_transformer_visits(self, X, y):
        """
        The projections and the final concat.
        """
        # pylint: disable=invalid-name
        self.column_transformer_visits_projections(X)
        result = self.transformer.fit_transform(X, y)
        self.column_transformer_visits_save_child_results()
        result = self.column_transformer_visits_concat(result)
        return result

    def column_transformer_visits_projections(self, X):
        """
        Inspection visits for the different projections
        """
        # pylint: disable=invalid-name, too-many-locals
        transformers_tuples = self.transformer.transformers
        columns_with_transformer = [(column, transformer_tuple[1]) for transformer_tuple in transformers_tuples
                                    for column in transformer_tuple[2]]
        for column, transformer in columns_with_transformer:
            projected_df = X[[column]]
            function_info = (self.module_name, "fit_transform")
            operator_context = OperatorContext(OperatorType.PROJECTION, function_info)
            description = "to ['{}'] (ColumnTransformer)".format(column)
            local_result = execute_inspection_visits_unary_op(operator_context, self.code_reference, X,
                                                              X.annotations, projected_df, self.inspections,
                                                              self.code_ref_inspection_output_map, description,
                                                              [column])

            # If the transformer is a column transformer, we have multiple annotations we need to pass to different
            # transformers.  If we do not want to override internal column transformer functions, we have to work around
            # these black box functions and pass the annotations using a different mechanism
            current_annotations = local_result.annotations
            current_annotations_for_transformer = transformer.annotation_result_project_workaround or []
            current_annotations_for_transformer.append(current_annotations)
            transformer.annotation_result_project_workaround = current_annotations_for_transformer

    def column_transformer_visits_save_child_results(self):
        """
        Because Column transformer creates deep copies, we need to extract results here
        """
        transformers_tuples = self.transformer.transformers_[:-1]
        transformers = [transformer_tuple[1] for transformer_tuple in transformers_tuples]
        for transformer in transformers:
            self.code_ref_inspection_output_map.update(transformer.code_ref_inspection_output_map)

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
            _, annotation = annotations[index]
            transformer_data_with_annotations.append((data, annotation))
        function_info = (self.module_name, "fit_transform")
        operator_context = OperatorContext(OperatorType.CONCATENATION, function_info)
        description = "concat"
        result = execute_inspection_visits_nary_op(operator_context, self.code_reference,
                                                   transformer_data_with_annotations, result, self.inspections,
                                                   self.code_ref_inspection_output_map, description)
        return result

    def estimator_visits(self, X, y):
        """
        Inspection visits for the estimator DAG node
        """
        # pylint: disable=invalid-name
        function_info = (self.module_name, "fit")
        assert y is not None
        operator_context = OperatorContext(OperatorType.ESTIMATOR, function_info)
        description = "fit"
        execute_inspection_visits_sink_op(operator_context, self.code_reference,
                                          X, y, self.inspections,
                                          self.code_ref_inspection_output_map, description)

    def train_data_and_labels_visits(self, X, y):
        """
        Pipeline.fit returns nothing (trained models get no edge in our DAG).
        Only need to do two scans for train data and train labels
        """
        # pylint: disable=invalid-name
        function_info = (self.module_name, "fit")
        operator_context = OperatorContext(OperatorType.TRAIN_DATA, function_info)
        X_annotated = execute_inspection_visits_unary_op(operator_context, self.code_reference, X, X.annotations, X,
                                                         self.inspections, self.code_ref_inspection_output_map, "fit X",
                                                         list(X.columns.values))
        assert y is not None
        operator_context = OperatorContext(OperatorType.TRAIN_LABELS, function_info)
        if isinstance(y, MlinspectSeries):
            columns = [y.name]
        else:
            columns = ["array"]
        y_annotated = execute_inspection_visits_unary_op(operator_context, self.code_reference, y,
                                                         y.annotations, y, self.inspections,
                                                         self.code_ref_inspection_output_map, "fit y", columns)
        return X_annotated, y_annotated

    def score(self, X, y):
        """
        Forward some score call of an estimator
        """
        # pylint: disable=invalid-name
        return self.transformer.score(X, y)


# -------------------------------------------------------
# Execute inspections functions
# -------------------------------------------------------

def execute_inspection_visits_nary_op(operator_context, code_reference, transformer_data_with_annotations,
                                      output_data, inspections, code_ref_inspection_output_map, func_name):
    """Execute inspections"""
    # pylint: disable=too-many-arguments
    inspection_count = len(inspections)
    iterators_for_inspections = iter_input_annotation_output_nary_op(inspection_count,
                                                                     transformer_data_with_annotations,
                                                                     output_data,
                                                                     operator_context)
    annotation_iterators = execute_visits(inspections, iterators_for_inspections)
    return_value = store_inspection_outputs(annotation_iterators, code_reference, output_data, inspections,
                                            code_ref_inspection_output_map, func_name, False, ["array"])
    return return_value


def execute_inspection_visits_sink_op(operator_context, code_reference, data, target,
                                      inspections, code_reference_inspection_output_map, func_name):
    """Execute inspections"""
    # pylint: disable=too-many-arguments
    assert isinstance(data, (MlinspectCsrMatrix, MlinspectNdarray, MlinspectDataFrame))
    assert isinstance(target, (MlinspectNdarray, MlinspectSeries, MlinspectDataFrame))
    inspection_count = len(inspections)
    iterators_for_inspections = iter_input_annotation_output_sink_op(inspection_count, data, target, operator_context)
    annotation_iterators = execute_visits(inspections, iterators_for_inspections)
    store_inspection_outputs(annotation_iterators, code_reference, None, inspections,
                             code_reference_inspection_output_map, func_name, True, None)


def execute_inspection_visits_unary_op(operator_context, code_reference, input_data, input_annotations, output_data,
                                       inspections, code_reference_inspection_output_map, func_name, columns):
    """Execute inspections"""
    # pylint: disable=too-many-arguments
    inspection_count = len(inspections)
    iterators_for_inspections = iter_input_annotation_output_map(inspection_count,
                                                                 input_data,
                                                                 input_annotations,
                                                                 output_data,
                                                                 operator_context,
                                                                 columns)
    annotation_iterators = execute_visits(inspections, iterators_for_inspections)
    return_value = store_inspection_outputs(annotation_iterators, code_reference, output_data, inspections,
                                            code_reference_inspection_output_map, func_name, False, columns)
    return return_value


def execute_visits(inspections, iterators_for_inspections):
    """
    After creating the iterators we need depending on the operator type, we need to execute the
    generic inspection visits
    """
    annotation_iterators = []
    for inspection_index, inspection in enumerate(inspections):
        iterator_for_inspection = iterators_for_inspections[inspection_index]
        annotations_iterator = inspection.visit_operator(iterator_for_inspection)
        annotation_iterators.append(annotations_iterator)
    return annotation_iterators


# -------------------------------------------------------
# Store inspection results functions
# -------------------------------------------------------

def store_inspection_outputs(annotation_iterators, code_reference, return_value, inspections,
                             code_reference_inspection_output_map, func_name, is_sink, columns):
    """
    Stores the inspection annotations for the rows in the dataframe and the
    inspection annotations for the DAG operators in a map
    """
    # pylint: disable=too-many-arguments, too-many-locals, too-many-branches
    annotations_ndarray = build_annotation_list_from_iters(annotation_iterators)
    inspection_outputs = {}
    for inspection in inspections:
        inspection_outputs[inspection] = inspection.get_operator_annotation_after_visit()

    stored_inspection_results = code_reference_inspection_output_map.get(code_reference, {})
    stored_inspection_results[func_name] = (inspection_outputs, columns)
    code_reference_inspection_output_map[code_reference] = stored_inspection_results

    if is_sink:
        new_return_value = None
    else:
        new_return_value = create_wrapper_with_annotations(annotations_ndarray, return_value)

    return new_return_value
