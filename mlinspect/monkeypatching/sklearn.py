"""
Monkey patching for sklearn
"""
from functools import partial

import gorilla
import numpy
from scipy.sparse import csr_matrix
from sklearn import preprocessing, compose, tree

from mlinspect.backends._pandas_backend import execute_inspection_visits_unary_operator
from mlinspect.backends._sklearn_backend import SklearnBackend
from mlinspect.inspections._inspection_input import OperatorContext
from mlinspect.instrumentation import _pipeline_executor
from mlinspect.instrumentation._dag_node import OperatorType, DagNode
from mlinspect.monkeypatching.monkey_patching_utils import execute_patched_func, add_dag_node, \
    execute_patched_func_indirect_allowed, get_input_info
from mlinspect.monkeypatching.numpy import MlinspectNdarray


@gorilla.patches(preprocessing)
class SklearnPreprocessingPatching:
    """ Patches for sklearn """

    # pylint: disable=too-few-public-methods

    @gorilla.name('label_binarize')
    @gorilla.settings(allow_hit=True)
    def patched_label_binarize(*args, **kwargs):
        """ Patch for ('sklearn.preprocessing._label', 'label_binarize') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(preprocessing, 'label_binarize')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            function_info = ('sklearn.preprocessing._label', 'label_binarize')
            input_info = get_input_info(args[0], caller_filename, lineno, function_info, optional_code_reference,
                                        optional_source_code)

            operator_context = OperatorContext(OperatorType.PROJECTION_MODIFY, function_info)
            input_infos = SklearnBackend.before_call(operator_context, [input_info.annotated_dfobject])
            result = original(input_infos[0].result_data, *args[1:], **kwargs)
            backend_result = SklearnBackend.after_call(operator_context,
                                                       input_infos,
                                                       result)

            classes = kwargs['classes']
            description = "label_binarize, classes: {}".format(classes)
            dag_node = DagNode(op_id, caller_filename, lineno, OperatorType.PROJECTION_MODIFY, function_info,
                               description,
                               ["array"], optional_code_reference, optional_source_code)
            add_dag_node(dag_node, [input_info.dag_node], backend_result)
            return result

        return execute_patched_func(original, execute_inspections, *args, **kwargs)


class SklearnCallInfo:
    """ Contains info like lineno from the current Transformer so indirect utility function calls can access it """
    # pylint: disable=too-few-public-methods

    transformer_op_id = None
    transformer_filename = None
    transformer_lineno = None
    module = None
    transformer_optional_code_reference = None
    transformer_optional_source_code = None
    column_transformer_active = False


call_info_singleton = SklearnCallInfo()


@gorilla.patches(compose.ColumnTransformer)
class SklearnComposePatching:
    """ Patches for sklearn ColumnTransformer"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self,
                        transformers, *,
                        remainder='drop',
                        sparse_threshold=0.3,
                        n_jobs=None,
                        transformer_weights=None,
                        verbose=False):
        """ Patch for ('sklearn.compose._column_transformer', 'ColumnTransformer') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(compose.ColumnTransformer, '__init__')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=attribute-defined-outside-init
            original(self, transformers, remainder=remainder, sparse_threshold=sparse_threshold, n_jobs=n_jobs,
                     transformer_weights=transformer_weights, verbose=verbose)

            self.mlinspect_op_id = op_id
            self.mlinspect_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_indirect_allowed(execute_inspections)

    @gorilla.name('fit_transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.compose._column_transformer', 'ColumnTransformer') """
        # pylint: disable=no-method-argument
        call_info_singleton.transformer_filename = self.mlinspect_filename
        call_info_singleton.transformer_lineno = self.mlinspect_lineno
        call_info_singleton.module = ('sklearn.compose._column_transformer', 'ColumnTransformer')
        call_info_singleton.transformer_optional_code_reference = self.mlinspect_optional_code_reference
        call_info_singleton.transformer_optional_source_code = self.mlinspect_optional_source_code

        call_info_singleton.column_transformer_active = True
        original = gorilla.get_original_attribute(compose.ColumnTransformer, 'fit_transform')
        result = original(self, *args, **kwargs)
        call_info_singleton.column_transformer_active = False

        return result

    @gorilla.name('_hstack')
    @gorilla.settings(allow_hit=True)
    def patched_hstack(self, *args, **kwargs):
        """ Patch for ('sklearn.compose._column_transformer', 'ColumnTransformer') """
        # pylint: disable=no-method-argument, unused-argument, too-many-locals
        original = gorilla.get_original_attribute(compose.ColumnTransformer, '_hstack')
        input_tuple = args[0]
        module = ('sklearn.compose._column_transformer', 'ColumnTransformer')
        input_infos = []
        for input_df_obj in input_tuple:
            input_info = get_input_info(input_df_obj, self.mlinspect_filename, self.mlinspect_lineno, module,
                                        self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)
            input_infos.append(input_info)

        user_operation = LogicalConcat(1, True, self.sparse_output_)  # pylint: disable=no-member
        engine_inputs = [input_info.engine_input for input_info in input_infos]
        fallback = partial(original, self, *args, **kwargs)
        engine_result = _pipeline_executor.singleton.engine.run(engine_inputs, user_operation, fallback)

        if self.sparse_output_:  # pylint: disable=no-member
            result = engine_result.user_op_result.to_csr_matrix()
        else:
            result = engine_result.user_op_result.to_numpy_2d_array()
            assert isinstance(result, MlinspectNdarray)

        dag_node = DagNode(self.mlinspect_op_id, self.mlinspect_filename, self.mlinspect_lineno,
                           OperatorType.CONCATENATION, module, "", ['array'], self.mlinspect_optional_code_reference,
                           self.mlinspect_optional_source_code)
        input_dag_nodes = [input_info.dag_node for input_info in input_infos]
        add_dag_node(dag_node, input_dag_nodes, result, engine_result)

        return result


@gorilla.patches(preprocessing.StandardScaler)
class SklearnStandardScalerPatching:
    """ Patches for sklearn StandardScaler"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *, copy=True, with_mean=True, with_std=True, mlinspect_op_id=None,
                        mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None):
        """ Patch for ('sklearn.preprocessing._data', 'StandardScaler') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(preprocessing.StandardScaler, '__init__')

        self.mlinspect_op_id = mlinspect_op_id
        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, copy=copy, with_mean=with_mean, with_std=with_std)

            self.mlinspect_op_id = op_id
            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func(original, execute_inspections, self, copy=copy, with_mean=with_mean,
                                    with_std=with_std)

    @gorilla.name('fit_transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.preprocessing._data.StandardScaler', 'fit_transform') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(preprocessing.StandardScaler, 'fit_transform')
        function_info = ('sklearn.preprocessing._data', 'StandardScaler')
        input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                    self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

        operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
        input_infos = SklearnBackend.before_call(operator_context, [input_info.annotated_dfobject])
        result = original(self, input_infos[0].result_data, *args[1:], **kwargs)
        backend_result = SklearnBackend.after_call(operator_context,
                                                   input_infos,
                                                   result)
        new_return_value = backend_result.annotated_dfobject.result_data
        assert isinstance(new_return_value, MlinspectNdarray)
        dag_node = DagNode(self.mlinspect_op_id, self.mlinspect_caller_filename, self.mlinspect_lineno,
                           OperatorType.TRANSFORMER, function_info, "Standard Scaler", ['array'],
                           self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)
        add_dag_node(dag_node, [input_info.dag_node], backend_result)
        return new_return_value


@gorilla.patches(preprocessing.OneHotEncoder)
class SklearnOneHotEncoderPatching:
    """ Patches for sklearn OneHotEncoder"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *, categories='auto', drop=None, sparse=True,
                        dtype=numpy.float64, handle_unknown='error', mlinspect_op_id=None,
                        mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None):
        """ Patch for ('sklearn.preprocessing._encoders', 'OneHotEncoder') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(preprocessing.OneHotEncoder, '__init__')

        self.mlinspect_op_id = mlinspect_op_id
        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, categories=categories, drop=drop, sparse=sparse, dtype=dtype, handle_unknown=handle_unknown)

            self.mlinspect_op_id = op_id
            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func(original, execute_inspections, self, categories=categories, drop=drop,
                                    sparse=sparse, dtype=dtype, handle_unknown=handle_unknown)

    @gorilla.name('fit_transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.preprocessing._encoders.OneHotEncoder', 'fit_transform') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(preprocessing.OneHotEncoder, 'fit_transform')
        module = ('sklearn.preprocessing._encoders', 'OneHotEncoder')
        input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, module,
                                    self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)
        partial_fit_transform = lambda x: original(self, x, *args[1:], **kwargs)
        input_type = DataframeType.get_dataframe_type(args[0])
        user_operation = LogicalProjectionTransformer(1, partial_fit_transform, module, input_type)
        engine_input = [input_info.engine_input]
        fallback = partial(original, self, *args, **kwargs)
        engine_result = _pipeline_executor.singleton.engine.run(engine_input, user_operation, fallback)
        if self.sparse:  # pylint: disable=no-member
            result = engine_result.user_op_result.to_csr_matrix()
            assert isinstance(result, csr_matrix)
        else:
            result = engine_result.user_op_result.to_numpy_2d_array()
            assert isinstance(result, MlinspectNdarray)
        dag_node = DagNode2(self.mlinspect_op_id, self.mlinspect_caller_filename, self.mlinspect_lineno,
                            OperatorType2.TRANSFORMER, module, "One-Hot Encoder", ['array'],
                            self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)
        add_dag_node(dag_node, [input_info.dag_node], result, engine_result)
        return result


@gorilla.patches(tree.DecisionTreeClassifier)
class SklearnDecisionTreePatching:
    """ Patches for sklearn DecisionTree"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *, criterion="gini", splitter="best", max_depth=None, min_samples_split=2,
                        min_samples_leaf=1, min_weight_fraction_leaf=0., max_features=None, random_state=None,
                        max_leaf_nodes=None, min_impurity_decrease=0., min_impurity_split=None, class_weight=None,
                        presort='deprecated', ccp_alpha=0.0, mlinspect_op_id=None, mlinspect_caller_filename=None,
                        mlinspect_lineno=None, mlinspect_optional_code_reference=None,
                        mlinspect_optional_source_code=None):
        """ Patch for ('sklearn.tree._classes', 'DecisionTreeClassifier') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init, too-many-locals
        original = gorilla.get_original_attribute(tree.DecisionTreeClassifier, '__init__')

        self.mlinspect_op_id = mlinspect_op_id
        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, criterion=criterion, splitter=splitter, max_depth=max_depth,
                     min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                     min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                     random_state=random_state, max_leaf_nodes=max_leaf_nodes,
                     min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split,
                     class_weight=class_weight, presort=presort, ccp_alpha=ccp_alpha)

            self.mlinspect_op_id = op_id
            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func(original, execute_inspections, self, criterion=criterion, splitter=splitter,
                                    max_depth=max_depth, min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                    random_state=random_state, max_leaf_nodes=max_leaf_nodes,
                                    min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split,
                                    class_weight=class_weight, presort=presort, ccp_alpha=ccp_alpha)

    @gorilla.name('fit')
    @gorilla.settings(allow_hit=True)
    def patched_fit(self, *args, **kwargs):
        """ Patch for ('sklearn.tree._classes.DecisionTreeClassifier', 'fit') """
        # pylint: disable=no-method-argument, too-many-locals
        original = gorilla.get_original_attribute(tree.DecisionTreeClassifier, 'fit')
        module = ('sklearn.tree._classes', 'DecisionTreeClassifier')

        input_info_train_data = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, module,
                                               self.mlinspect_optional_code_reference,
                                               self.mlinspect_optional_source_code)

        train_data_op_id = _pipeline_executor.singleton.get_next_op_id()
        train_data_dag_node = DagNode2(train_data_op_id, self.mlinspect_caller_filename,
                                       self.mlinspect_lineno, OperatorType2.TRAIN_DATA, module, "Train Data", ["array"],
                                       self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)
        train_data_operation = LogicalNoOp(1)
        engine_input = [input_info_train_data.engine_input]
        fallback = lambda: args[0]
        engine_result_data = _pipeline_executor.singleton.engine.run(engine_input, train_data_operation, fallback)
        add_dag_node(train_data_dag_node, [input_info_train_data.dag_node], args[0], engine_result_data)

        input_info_train_labels = get_input_info(args[1], self.mlinspect_caller_filename, self.mlinspect_lineno, module,
                                                 self.mlinspect_optional_code_reference,
                                                 self.mlinspect_optional_source_code)
        train_label_op_id = _pipeline_executor.singleton.get_next_op_id()
        train_labels_dag_node = DagNode2(train_label_op_id, self.mlinspect_caller_filename, self.mlinspect_lineno,
                                         OperatorType2.TRAIN_LABELS, module, "Train Labels", ["array"],
                                         self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)
        train_data_operation = LogicalNoOp(1)
        engine_input = [input_info_train_labels.engine_input]
        fallback = lambda: args[1]
        engine_result_labels = _pipeline_executor.singleton.engine.run(engine_input, train_data_operation, fallback)
        add_dag_node(train_labels_dag_node, [input_info_train_labels.dag_node], args[1], engine_result_labels)

        engine_input = [get_engine_input(engine_result_data), get_engine_input(engine_result_labels)]
        partial_fit = lambda x, y: original(self, x, y, *args[2:], **kwargs)
        data_input_type = DataframeType.get_dataframe_type(args[0])
        label_input_type = DataframeType.get_dataframe_type(args[1])
        estimator_operation = LogicalEstimator(1, partial_fit, data_input_type, label_input_type)
        fallback = partial(original, self, *args, **kwargs)
        engine_result_estimator = _pipeline_executor.singleton.engine.run(engine_input, estimator_operation, fallback)

        dag_node = DagNode2(self.mlinspect_op_id, self.mlinspect_caller_filename, self.mlinspect_lineno,
                            OperatorType2.ESTIMATOR, module, "Decision Tree", [],
                            self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)
        add_dag_node(dag_node, [train_data_dag_node, train_labels_dag_node], None, engine_result_estimator)
        return self
