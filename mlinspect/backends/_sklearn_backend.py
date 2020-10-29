"""
The scikit-learn backend
"""

import networkx
from sklearn.base import BaseEstimator
from tensorflow.python.keras.wrappers.scikit_learn import BaseWrapper

from ._backend import Backend
from ._pandas_backend import execute_inspection_visits_unary_operator
from ._pandas_backend_frame_wrapper import MlinspectSeries, MlinspectDataFrame
from ._sklearn_backend_transformer_wrapper import MlinspectEstimatorTransformer, transformer_names
from ._sklearn_dag_processor import SklearnDagPostprocessor
from ._sklearn_wir_processor import SklearnWirPreprocessor
from ..inspections._inspection_input import OperatorContext
from ..instrumentation._dag_node import OperatorType


class SklearnBackend(Backend):
    """
    The scikit-learn backend
    """

    operator_map = {
        ('sklearn.preprocessing._label', 'label_binarize'): OperatorType.PROJECTION_MODIFY,
        ('sklearn.compose._column_transformer', 'ColumnTransformer', 'Projection'): OperatorType.PROJECTION,
        ('sklearn.preprocessing._encoders', 'OneHotEncoder', 'Pipeline'): OperatorType.TRANSFORMER,
        ('sklearn.preprocessing._data', 'StandardScaler', 'Pipeline'): OperatorType.TRANSFORMER,
        ('sklearn.impute._base', 'SimpleImputer', 'Pipeline'): OperatorType.TRANSFORMER,
        ('sklearn.preprocessing._discretization', 'KBinsDiscretizer', 'Pipeline'): OperatorType.TRANSFORMER,
        ('sklearn.compose._column_transformer', 'ColumnTransformer', 'Concatenation'): OperatorType.CONCATENATION,
        ('sklearn.tree._classes', 'DecisionTreeClassifier', 'Pipeline'): OperatorType.ESTIMATOR,
        ('sklearn.linear_model._logistic', 'LogisticRegression', 'Pipeline'): OperatorType.ESTIMATOR,
        ('sklearn.pipeline', 'fit', 'Pipeline'): OperatorType.FIT,
        ('sklearn.pipeline', 'fit', 'Train Data'): OperatorType.TRAIN_DATA,
        ('sklearn.pipeline', 'fit', 'Train Labels'): OperatorType.TRAIN_LABELS,
        ('sklearn.model_selection._split', 'train_test_split'): OperatorType.TRAIN_TEST_SPLIT,
        # TODO: We  can remove this later by checking if subclass of transformer/estimator
        ('example_pipelines.healthcare.healthcare_utils', 'MyW2VTransformer', 'Pipeline'): OperatorType.TRANSFORMER,
        ('tensorflow.python.keras.wrappers.scikit_learn', 'KerasClassifier', 'Pipeline'): OperatorType.ESTIMATOR
    }

    def is_responsible_for_call(self, function_info, function_prefix, value=None):
        """Checks whether the backend is responsible for the current method call"""
        return function_prefix == "sklearn" or isinstance(value, (BaseEstimator, BaseWrapper))

    def process_wir(self, wir: networkx.DiGraph) -> networkx.DiGraph:
        """
        Preprocess scikit-learn pipeline operations to hide the special pipeline
        declaration style from other parts of the library
        """
        return SklearnWirPreprocessor().process_wir(wir)

    def process_dag(self, dag: networkx.DiGraph) -> networkx.DiGraph:
        """
        Preprocess scikit-learn pipeline operations to hide the special pipeline
        declaration style from other parts of the library
        """
        post_processor_result = SklearnDagPostprocessor() \
            .process_dag(dag, self.wir_post_processing_map)
        new_dag_node_identifier_to_inspection_output = post_processor_result[0]
        new_dag_node_identifier_to_columns = post_processor_result[1]

        self.dag_node_identifier_to_inspection_output = {**self.dag_node_identifier_to_inspection_output,
                                                         **new_dag_node_identifier_to_inspection_output}
        self.dag_node_identifier_to_columns = {**self.dag_node_identifier_to_columns,
                                               **new_dag_node_identifier_to_columns}
        return dag

    def __init__(self):
        super().__init__()
        self.input_data = None
        self.wir_post_processing_map = {}

    def before_call_used_value(self, function_info, subscript, call_code, value_code, value_value,
                               code_reference):
        """The value or module a function may be called on"""
        # pylint: disable=too-many-arguments, unused-argument, no-self-use, unnecessary-pass
        if isinstance(value_value, (BaseEstimator, BaseWrapper)):
            self.input_data = value_value

    def before_call_used_args(self, function_info, subscript, call_code, args_code, code_reference, store, args_values):
        """The arguments a function may be called with"""
        # pylint: disable=too-many-arguments, unused-argument, no-self-use
        if function_info == ('sklearn.preprocessing._label', 'label_binarize'):
            assert isinstance(args_values[0], MlinspectSeries)
            self.input_data = args_values[0]
        elif function_info == ('sklearn.model_selection._split', 'train_test_split'):
            assert isinstance(args_values[0], MlinspectDataFrame)
            args_values[0]['mlinspect_index'] = range(0, len(args_values[0]))
            self.input_data = args_values[0]

    def before_call_used_kwargs(self, function_info, subscript, call_code, kwargs_code, code_reference, kwargs_values):
        """The keyword arguments a function may be called with"""
        # pylint: disable=too-many-arguments, unused-argument, no-self-use
        description = None
        if function_info == ('sklearn.preprocessing._label', 'label_binarize'):
            classes = kwargs_values['classes']
            description = "label_binarize, classes: {}".format(classes)
        elif function_info == ('sklearn.model_selection._split', 'train_test_split'):
            description = "(Train Data)"

        if description:
            self.code_reference_to_description[code_reference] = description

    def after_call_used(self, function_info, subscript, call_code, return_value, code_reference):
        """The return value of some function"""
        # pylint: disable=too-many-arguments, unused-argument, no-self-use
        self.save_call_module_and_description(code_reference, function_info, self.input_data)

        if function_info == ('sklearn.preprocessing._label', 'label_binarize'):
            operator_context = OperatorContext(OperatorType.PROJECTION_MODIFY, function_info)
            return_value = execute_inspection_visits_unary_operator(self, operator_context, code_reference,
                                                                    self.input_data,
                                                                    self.input_data.annotations,
                                                                    return_value,
                                                                    False)
        elif function_info == ('sklearn.model_selection._split', 'train_test_split'):
            operator_context = OperatorContext(OperatorType.TRAIN_TEST_SPLIT, function_info)
            train_data, test_data = return_value
            train_data = execute_inspection_visits_unary_operator(self, operator_context, code_reference,
                                                                  self.input_data,
                                                                  self.input_data.annotations,
                                                                  train_data,
                                                                  True)
            return_value = train_data, test_data
        elif function_info in {('sklearn.preprocessing._encoders', 'OneHotEncoder'),
                               ('sklearn.preprocessing._data', 'StandardScaler'),
                               ('sklearn.tree._classes', 'DecisionTreeClassifier'),
                               ('tensorflow.python.keras.wrappers.scikit_learn', 'KerasClassifier'),
                               ('example_pipelines.healthcare.healthcare_utils', 'MyW2VTransformer'),
                               ('sklearn.impute._base', 'SimpleImputer'),
                               ('sklearn.compose._column_transformer', 'ColumnTransformer'),
                               ('sklearn.pipeline', 'Pipeline'),
                               ('sklearn.preprocessing._discretization', 'KBinsDiscretizer'),
                               ('sklearn.linear_model._logistic', 'LogisticRegression')
                               }:
            return_value = MlinspectEstimatorTransformer(return_value, code_reference, self.inspections,
                                                         self.wir_post_processing_map)

        self.input_data = None

        return return_value

    def save_call_module_and_description(self, code_reference, function_info, maybe_wrapper_transformer):
        """Replace the module of mlinspect transformer wrappers with the original modules"""
        if maybe_wrapper_transformer is not None and \
                function_info[0] == 'mlinspect.backends._sklearn_backend_transformer_wrapper' and \
                function_info[1] != "score":
            function_info = (maybe_wrapper_transformer.module_name, function_info[1])

        description = transformer_names.get(function_info, None)
        if description:
            self.code_reference_to_description[code_reference] = description

        self.code_reference_to_module[code_reference] = function_info
