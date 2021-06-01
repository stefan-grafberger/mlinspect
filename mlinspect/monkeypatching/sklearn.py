"""
Monkey patching for sklearn
"""

import gorilla
import numpy
from sklearn import preprocessing, compose, tree, impute

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
            new_return_value = backend_result.annotated_dfobject.result_data

            classes = kwargs['classes']
            description = "label_binarize, classes: {}".format(classes)
            dag_node = DagNode(op_id, caller_filename, lineno, OperatorType.PROJECTION_MODIFY, function_info,
                               description,
                               ["array"], optional_code_reference, optional_source_code)
            add_dag_node(dag_node, [input_info.dag_node], backend_result)

            return new_return_value

        return execute_patched_func(original, execute_inspections, *args, **kwargs)


class SklearnCallInfo:
    """ Contains info like lineno from the current Transformer so indirect utility function calls can access it """
    # pylint: disable=too-few-public-methods

    transformer_op_id = None
    transformer_filename = None
    transformer_lineno = None
    function_info = None
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
        call_info_singleton.function_info = ('sklearn.compose._column_transformer', 'ColumnTransformer')
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
        function_info = ('sklearn.compose._column_transformer', 'ColumnTransformer')
        input_infos = []
        for input_df_obj in input_tuple:
            input_info = get_input_info(input_df_obj, self.mlinspect_filename, self.mlinspect_lineno, function_info,
                                        self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)
            input_infos.append(input_info)

        operator_context = OperatorContext(OperatorType.CONCATENATION, function_info)
        input_annotated_dfs = [input_info.annotated_dfobject for input_info in input_infos]
        backend_input_infos = SklearnBackend.before_call(operator_context, input_annotated_dfs)
        # No input_infos copy needed because it's only a selection and the rows not being removed don't change
        result = original(self, *args, **kwargs)
        backend_result = SklearnBackend.after_call(operator_context,
                                                   backend_input_infos,
                                                   result)
        result = backend_result.annotated_dfobject.result_data

        dag_node = DagNode(self.mlinspect_op_id, self.mlinspect_filename, self.mlinspect_lineno,
                           OperatorType.CONCATENATION, function_info, "", ['array'],
                           self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)
        input_dag_nodes = [input_info.dag_node for input_info in input_infos]
        add_dag_node(dag_node, input_dag_nodes, backend_result)

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


@gorilla.patches(preprocessing.KBinsDiscretizer)
class SklearnKBinsDiscretizerPatching:
    """ Patches for sklearn KBinsDiscretizer"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, n_bins=5, *, encode='onehot', strategy='quantile', mlinspect_op_id=None,
                        mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None):
        """ Patch for ('sklearn.preprocessing._discretization', 'KBinsDiscretizer') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(preprocessing.KBinsDiscretizer, '__init__')

        self.mlinspect_op_id = mlinspect_op_id
        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, n_bins=n_bins, encode=encode, strategy=strategy)

            self.mlinspect_op_id = op_id
            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func(original, execute_inspections, self, n_bins=n_bins, encode=encode,
                                    strategy=strategy)

    @gorilla.name('fit_transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.preprocessing._discretization.KBinsDiscretizer', 'fit_transform') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(preprocessing.KBinsDiscretizer, 'fit_transform')
        function_info = ('sklearn.preprocessing._discretization', 'KBinsDiscretizer')
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
                           OperatorType.TRANSFORMER, function_info, "K-Bins Discretizer", ['array'],
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
        function_info = ('sklearn.preprocessing._encoders', 'OneHotEncoder')
        input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                    self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

        operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
        input_infos = SklearnBackend.before_call(operator_context, [input_info.annotated_dfobject])
        result = original(self, input_infos[0].result_data, *args[1:], **kwargs)
        backend_result = SklearnBackend.after_call(operator_context,
                                                   input_infos,
                                                   result)
        new_return_value = backend_result.annotated_dfobject.result_data
        dag_node = DagNode(self.mlinspect_op_id, self.mlinspect_caller_filename, self.mlinspect_lineno,
                           OperatorType.TRANSFORMER, function_info, "One-Hot Encoder", ['array'],
                           self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)
        add_dag_node(dag_node, [input_info.dag_node], backend_result)
        return new_return_value


@gorilla.patches(impute.SimpleImputer)
class SklearnSimpleImputerPatching:
    """ Patches for sklearn SimpleImputer"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *, missing_values=numpy.nan, strategy="mean",
                        fill_value=None, verbose=0, copy=True, add_indicator=False, mlinspect_op_id=None,
                        mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None):
        """ Patch for ('sklearn.impute._base', 'SimpleImputer') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(impute.SimpleImputer, '__init__')

        self.mlinspect_op_id = mlinspect_op_id
        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, missing_values=missing_values, strategy=strategy, fill_value=fill_value, verbose=verbose,
                     copy=copy, add_indicator=add_indicator)

            self.mlinspect_op_id = op_id
            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func(original, execute_inspections, self, missing_values=missing_values,
                                    strategy=strategy, fill_value=fill_value, verbose=verbose, copy=copy,
                                    add_indicator=add_indicator)

    @gorilla.name('fit_transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.preprocessing._encoders.OneHotEncoder', 'fit_transform') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(impute.SimpleImputer, 'fit_transform')
        function_info = ('sklearn.impute._base', 'SimpleImputer')
        input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                    self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

        operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
        input_infos = SklearnBackend.before_call(operator_context, [input_info.annotated_dfobject])
        result = original(self, input_infos[0].result_data, *args[1:], **kwargs)
        backend_result = SklearnBackend.after_call(operator_context,
                                                   input_infos,
                                                   result)
        new_return_value = backend_result.annotated_dfobject.result_data
        dag_node = DagNode(self.mlinspect_op_id, self.mlinspect_caller_filename, self.mlinspect_lineno,
                           OperatorType.TRANSFORMER, function_info, "Simple Imputer", ['array'],
                           self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)
        add_dag_node(dag_node, [input_info.dag_node], backend_result)
        return new_return_value


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
        function_info = ('sklearn.tree._classes', 'DecisionTreeClassifier')

        # Train data
        input_info_train_data = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno,
                                               function_info, self.mlinspect_optional_code_reference,
                                               self.mlinspect_optional_source_code)
        train_data_op_id = _pipeline_executor.singleton.get_next_op_id()
        train_data_dag_node = DagNode(train_data_op_id, self.mlinspect_caller_filename,
                                      self.mlinspect_lineno, OperatorType.TRAIN_DATA, function_info, "Train Data",
                                      ["array"], self.mlinspect_optional_code_reference,
                                      self.mlinspect_optional_source_code)
        operator_context = OperatorContext(OperatorType.TRAIN_DATA, function_info)
        input_infos = SklearnBackend.before_call(operator_context, [input_info_train_data.annotated_dfobject])
        data_backend_result = SklearnBackend.after_call(operator_context,
                                                        input_infos,
                                                        args[0])
        add_dag_node(train_data_dag_node, [input_info_train_data.dag_node], data_backend_result)
        train_data_result = data_backend_result.annotated_dfobject.result_data

        # Train labels
        operator_context = OperatorContext(OperatorType.TRAIN_LABELS, function_info)
        input_info_train_labels = get_input_info(args[1], self.mlinspect_caller_filename, self.mlinspect_lineno,
                                                 function_info, self.mlinspect_optional_code_reference,
                                                 self.mlinspect_optional_source_code)
        train_label_op_id = _pipeline_executor.singleton.get_next_op_id()
        train_labels_dag_node = DagNode(train_label_op_id, self.mlinspect_caller_filename, self.mlinspect_lineno,
                                        OperatorType.TRAIN_LABELS, function_info, "Train Labels", ["array"],
                                        self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)
        input_infos = SklearnBackend.before_call(operator_context, [input_info_train_labels.annotated_dfobject])
        label_backend_result = SklearnBackend.after_call(operator_context,
                                                         input_infos,
                                                         args[1])
        add_dag_node(train_labels_dag_node, [input_info_train_labels.dag_node], label_backend_result)
        train_labels_result = label_backend_result.annotated_dfobject.result_data

        # Estimator
        operator_context = OperatorContext(OperatorType.ESTIMATOR, function_info)
        input_dfs = [data_backend_result.annotated_dfobject, label_backend_result.annotated_dfobject]
        input_infos = SklearnBackend.before_call(operator_context, input_dfs)
        original(self, train_data_result, train_labels_result, *args[2:], **kwargs)
        estimator_backend_result = SklearnBackend.after_call(operator_context,
                                                             input_infos,
                                                             None)

        dag_node = DagNode(self.mlinspect_op_id, self.mlinspect_caller_filename, self.mlinspect_lineno,
                           OperatorType.ESTIMATOR, function_info, "Decision Tree", [],
                           self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)
        add_dag_node(dag_node, [train_data_dag_node, train_labels_dag_node], estimator_backend_result)
        return self
