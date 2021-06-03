"""
Monkey patching for sklearn
"""

import gorilla
import numpy
import pandas
from sklearn import preprocessing, compose, tree, impute, linear_model, model_selection
from tensorflow.keras.wrappers import scikit_learn as keras_sklearn_external  # pylint: disable=no-name-in-module
from tensorflow.python.keras.wrappers import scikit_learn as keras_sklearn_internal  # pylint: disable=no-name-in-module

from mlinspect.backends._backend import BackendResult
from mlinspect.backends._sklearn_backend import SklearnBackend
from mlinspect.inspections._inspection_input import OperatorContext, FunctionInfo, OperatorType
from mlinspect.instrumentation import _pipeline_executor
from mlinspect.instrumentation._dag_node import DagNode, BasicCodeLocation, DagNodeDetails, CodeReference
from mlinspect.instrumentation._pipeline_executor import singleton
from mlinspect.monkeypatching._monkey_patching_utils import execute_patched_func, add_dag_node, \
    execute_patched_func_indirect_allowed, get_input_info, execute_patched_func_no_op_id, get_optional_code_info_or_none
from mlinspect.monkeypatching._patch_numpy import MlinspectNdarray


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
            function_info = FunctionInfo('sklearn.preprocessing._label', 'label_binarize')
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
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, ["array"]),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            add_dag_node(dag_node, [input_info.dag_node], backend_result)

            return new_return_value

        return execute_patched_func(original, execute_inspections, *args, **kwargs)


@gorilla.patches(model_selection)
class SklearnModelSelectionPatching:
    """ Patches for sklearn """

    # pylint: disable=too-few-public-methods

    @gorilla.name('train_test_split')
    @gorilla.settings(allow_hit=True)
    def patched_train_test_split(*args, **kwargs):
        """ Patch for ('sklearn.model_selection._split', 'train_test_split') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(model_selection, 'train_test_split')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            function_info = FunctionInfo('sklearn.model_selection._split', 'train_test_split')
            input_info = get_input_info(args[0], caller_filename, lineno, function_info, optional_code_reference,
                                        optional_source_code)

            operator_context = OperatorContext(OperatorType.TRAIN_TEST_SPLIT, function_info)
            input_infos = SklearnBackend.before_call(operator_context, [input_info.annotated_dfobject])
            result = original(input_infos[0].result_data, *args[1:], **kwargs)
            backend_result = SklearnBackend.after_call(operator_context,
                                                       input_infos,
                                                       result)  # We ignore the test set for now
            train_backend_result = BackendResult(backend_result.annotated_dfobject,
                                                 backend_result.dag_node_annotation)
            test_backend_result = BackendResult(backend_result.optional_second_annotated_dfobject,
                                                backend_result.optional_second_dag_node_annotation)

            description = "(Train Data)"
            columns = list(result[0].columns)
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, columns),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            add_dag_node(dag_node, [input_info.dag_node], train_backend_result)

            description = "(Test Data)"
            columns = list(result[1].columns)
            dag_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, columns),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            add_dag_node(dag_node, [input_info.dag_node], test_backend_result)

            new_return_value = (train_backend_result.annotated_dfobject.result_data,
                                test_backend_result.annotated_dfobject.result_data)

            return new_return_value

        return execute_patched_func(original, execute_inspections, *args, **kwargs)


class SklearnCallInfo:
    """ Contains info like lineno from the current Transformer so indirect utility function calls can access it """
    # pylint: disable=too-few-public-methods

    transformer_filename: str or None = None
    transformer_lineno: int or None = None
    transformer_function_info: FunctionInfo or None = None
    transformer_optional_code_reference: CodeReference or None = None
    transformer_optional_source_code: str or None = None
    column_transformer_active: bool = False


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

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=attribute-defined-outside-init
            original(self, transformers, remainder=remainder, sparse_threshold=sparse_threshold, n_jobs=n_jobs,
                     transformer_weights=transformer_weights, verbose=verbose)

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
        call_info_singleton.transformer_function_info = FunctionInfo('sklearn.compose._column_transformer',
                                                                     'ColumnTransformer')
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

        if not call_info_singleton.column_transformer_active:
            return original(self, *args, **kwargs)

        input_tuple = args[0]
        function_info = FunctionInfo('sklearn.compose._column_transformer', 'ColumnTransformer')
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

        dag_node = DagNode(singleton.get_next_op_id(),
                           BasicCodeLocation(self.mlinspect_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails(None, ['array']),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code))
        input_dag_nodes = [input_info.dag_node for input_info in input_infos]
        add_dag_node(dag_node, input_dag_nodes, backend_result)

        return result


@gorilla.patches(preprocessing.StandardScaler)
class SklearnStandardScalerPatching:
    """ Patches for sklearn StandardScaler"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *, copy=True, with_mean=True, with_std=True,
                        mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None):
        """ Patch for ('sklearn.preprocessing._data', 'StandardScaler') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(preprocessing.StandardScaler, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, copy=copy, with_mean=with_mean, with_std=with_std)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self, copy=copy, with_mean=with_mean,
                                             with_std=with_std)

    @gorilla.name('fit_transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.preprocessing._data.StandardScaler', 'fit_transform') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(preprocessing.StandardScaler, 'fit_transform')
        function_info = FunctionInfo('sklearn.preprocessing._data', 'StandardScaler')
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
        dag_node = DagNode(singleton.get_next_op_id(),
                           BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails("Standard Scaler", ['array']),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code))
        add_dag_node(dag_node, [input_info.dag_node], backend_result)
        return new_return_value


@gorilla.patches(preprocessing.KBinsDiscretizer)
class SklearnKBinsDiscretizerPatching:
    """ Patches for sklearn KBinsDiscretizer"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, n_bins=5, *, encode='onehot', strategy='quantile',
                        mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None):
        """ Patch for ('sklearn.preprocessing._discretization', 'KBinsDiscretizer') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(preprocessing.KBinsDiscretizer, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, n_bins=n_bins, encode=encode, strategy=strategy)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self, n_bins=n_bins, encode=encode,
                                             strategy=strategy)

    @gorilla.name('fit_transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.preprocessing._discretization.KBinsDiscretizer', 'fit_transform') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(preprocessing.KBinsDiscretizer, 'fit_transform')
        function_info = FunctionInfo('sklearn.preprocessing._discretization', 'KBinsDiscretizer')
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
        dag_node = DagNode(singleton.get_next_op_id(),
                           BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails("K-Bins Discretizer", ['array']),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code))
        add_dag_node(dag_node, [input_info.dag_node], backend_result)
        return new_return_value


@gorilla.patches(preprocessing.OneHotEncoder)
class SklearnOneHotEncoderPatching:
    """ Patches for sklearn OneHotEncoder"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *, categories='auto', drop=None, sparse=True,
                        dtype=numpy.float64, handle_unknown='error',
                        mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None):
        """ Patch for ('sklearn.preprocessing._encoders', 'OneHotEncoder') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(preprocessing.OneHotEncoder, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, categories=categories, drop=drop, sparse=sparse, dtype=dtype, handle_unknown=handle_unknown)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self, categories=categories, drop=drop,
                                             sparse=sparse, dtype=dtype, handle_unknown=handle_unknown)

    @gorilla.name('fit_transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.preprocessing._encoders.OneHotEncoder', 'fit_transform') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(preprocessing.OneHotEncoder, 'fit_transform')
        function_info = FunctionInfo('sklearn.preprocessing._encoders', 'OneHotEncoder')
        input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                    self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

        operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
        input_infos = SklearnBackend.before_call(operator_context, [input_info.annotated_dfobject])
        result = original(self, input_infos[0].result_data, *args[1:], **kwargs)
        backend_result = SklearnBackend.after_call(operator_context,
                                                   input_infos,
                                                   result)
        new_return_value = backend_result.annotated_dfobject.result_data
        dag_node = DagNode(singleton.get_next_op_id(),
                           BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails("One-Hot Encoder", ['array']),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code))
        add_dag_node(dag_node, [input_info.dag_node], backend_result)
        return new_return_value


@gorilla.patches(impute.SimpleImputer)
class SklearnSimpleImputerPatching:
    """ Patches for sklearn SimpleImputer"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *, missing_values=numpy.nan, strategy="mean",
                        fill_value=None, verbose=0, copy=True, add_indicator=False,
                        mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None):
        """ Patch for ('sklearn.impute._base', 'SimpleImputer') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(impute.SimpleImputer, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, missing_values=missing_values, strategy=strategy, fill_value=fill_value, verbose=verbose,
                     copy=copy, add_indicator=add_indicator)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self, missing_values=missing_values,
                                             strategy=strategy, fill_value=fill_value, verbose=verbose, copy=copy,
                                             add_indicator=add_indicator)

    @gorilla.name('fit_transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.preprocessing._encoders.OneHotEncoder', 'fit_transform') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(impute.SimpleImputer, 'fit_transform')
        function_info = FunctionInfo('sklearn.impute._base', 'SimpleImputer')
        input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                    self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

        operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
        input_infos = SklearnBackend.before_call(operator_context, [input_info.annotated_dfobject])
        result = original(self, input_infos[0].result_data, *args[1:], **kwargs)
        backend_result = SklearnBackend.after_call(operator_context,
                                                   input_infos,
                                                   result)
        new_return_value = backend_result.annotated_dfobject.result_data
        if isinstance(input_infos[0].result_data, pandas.DataFrame):
            columns = list(input_infos[0].result_data.columns)
        else:
            columns = ['array']

        dag_node = DagNode(singleton.get_next_op_id(),
                           BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails("Simple Imputer", columns),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code))
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
                        presort='deprecated', ccp_alpha=0.0, mlinspect_caller_filename=None,
                        mlinspect_lineno=None, mlinspect_optional_code_reference=None,
                        mlinspect_optional_source_code=None):
        """ Patch for ('sklearn.tree._classes', 'DecisionTreeClassifier') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init, too-many-locals
        original = gorilla.get_original_attribute(tree.DecisionTreeClassifier, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, criterion=criterion, splitter=splitter, max_depth=max_depth,
                     min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                     min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                     random_state=random_state, max_leaf_nodes=max_leaf_nodes,
                     min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split,
                     class_weight=class_weight, presort=presort, ccp_alpha=ccp_alpha)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self, criterion=criterion,
                                             splitter=splitter, max_depth=max_depth,
                                             min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                             min_weight_fraction_leaf=min_weight_fraction_leaf,
                                             max_features=max_features,
                                             random_state=random_state, max_leaf_nodes=max_leaf_nodes,
                                             min_impurity_decrease=min_impurity_decrease,
                                             min_impurity_split=min_impurity_split,
                                             class_weight=class_weight, presort=presort, ccp_alpha=ccp_alpha)

    @gorilla.name('fit')
    @gorilla.settings(allow_hit=True)
    def patched_fit(self, *args, **kwargs):
        """ Patch for ('sklearn.tree._classes.DecisionTreeClassifier', 'fit') """
        # pylint: disable=no-method-argument, too-many-locals
        original = gorilla.get_original_attribute(tree.DecisionTreeClassifier, 'fit')
        function_info = FunctionInfo('sklearn.tree._classes', 'DecisionTreeClassifier')

        # Train data
        input_info_train_data = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno,
                                               function_info, self.mlinspect_optional_code_reference,
                                               self.mlinspect_optional_source_code)
        train_data_op_id = _pipeline_executor.singleton.get_next_op_id()
        operator_context = OperatorContext(OperatorType.TRAIN_DATA, function_info)
        train_data_dag_node = DagNode(train_data_op_id,
                                      BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                                      operator_context,
                                      DagNodeDetails("Train Data", ["array"]),
                                      get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                                     self.mlinspect_optional_source_code))
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
        train_labels_dag_node = DagNode(train_label_op_id,
                                        BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                                        operator_context,
                                        DagNodeDetails("Train Labels", ["array"]),
                                        get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                                       self.mlinspect_optional_source_code))
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

        dag_node = DagNode(singleton.get_next_op_id(),
                           BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails("Decision Tree", []),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code))
        add_dag_node(dag_node, [train_data_dag_node, train_labels_dag_node], estimator_backend_result)
        return self


@gorilla.patches(linear_model.LogisticRegression)
class SklearnLogisticRegressionPatching:
    """ Patches for sklearn LogisticRegression"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, penalty='l2', *, dual=False, tol=1e-4, C=1.0,  # pylint: disable=invalid-name
                        fit_intercept=True, intercept_scaling=1, class_weight=None,
                        random_state=None, solver='lbfgs', max_iter=100,
                        multi_class='auto', verbose=0, warm_start=False, n_jobs=None,
                        l1_ratio=None, mlinspect_caller_filename=None,
                        mlinspect_lineno=None, mlinspect_optional_code_reference=None,
                        mlinspect_optional_source_code=None):
        """ Patch for ('sklearn.linear_model._logistic', 'LogisticRegression') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init, too-many-locals
        original = gorilla.get_original_attribute(linear_model.LogisticRegression, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, penalty=penalty, dual=dual, tol=tol, C=C,
                     fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
                     random_state=random_state, solver=solver, max_iter=max_iter,
                     multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs,
                     l1_ratio=l1_ratio)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self, penalty=penalty, dual=dual, tol=tol,
                                             C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                                             class_weight=class_weight,
                                             random_state=random_state, solver=solver, max_iter=max_iter,
                                             multi_class=multi_class, verbose=verbose, warm_start=warm_start,
                                             n_jobs=n_jobs,
                                             l1_ratio=l1_ratio)

    @gorilla.name('fit')
    @gorilla.settings(allow_hit=True)
    def patched_fit(self, *args, **kwargs):
        """ Patch for ('sklearn.linear_model._logistic.LogisticRegression', 'fit') """
        # pylint: disable=no-method-argument, too-many-locals
        original = gorilla.get_original_attribute(linear_model.LogisticRegression, 'fit')
        function_info = FunctionInfo('sklearn.linear_model._logistic', 'LogisticRegression')

        # Train data
        input_info_train_data = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno,
                                               function_info, self.mlinspect_optional_code_reference,
                                               self.mlinspect_optional_source_code)
        train_data_op_id = _pipeline_executor.singleton.get_next_op_id()
        operator_context = OperatorContext(OperatorType.TRAIN_DATA, function_info)
        train_data_dag_node = DagNode(train_data_op_id,
                                      BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                                      operator_context,
                                      DagNodeDetails("Train Data", ["array"]),
                                      get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                                     self.mlinspect_optional_source_code))
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
        train_labels_dag_node = DagNode(train_label_op_id,
                                        BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                                        operator_context,
                                        DagNodeDetails("Train Labels", ["array"]),
                                        get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                                       self.mlinspect_optional_source_code))
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

        dag_node = DagNode(singleton.get_next_op_id(),
                           BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails("Logistic Regression", []),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code))
        add_dag_node(dag_node, [train_data_dag_node, train_labels_dag_node], estimator_backend_result)
        return self


class SklearnKerasClassifierPatching:
    """ Patches for tensorflow KerasClassifier"""

    # pylint: disable=too-few-public-methods
    @gorilla.patch(keras_sklearn_internal.BaseWrapper, name='__init__', settings=gorilla.Settings(allow_hit=True))
    def patched__init__(self, build_fn=None, mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None,
                        **sk_params):
        """ Patch for ('tensorflow.python.keras.wrappers.scikit_learn', 'KerasClassifier') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init, too-many-locals, too-many-arguments
        original = gorilla.get_original_attribute(keras_sklearn_internal.BaseWrapper, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, build_fn=build_fn, **sk_params)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self, build_fn=build_fn, **sk_params)

    @gorilla.patch(keras_sklearn_external.KerasClassifier, name='fit', settings=gorilla.Settings(allow_hit=True))
    def patched_fit(self, *args, **kwargs):
        """ Patch for ('tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier', 'fit') """
        # pylint: disable=no-method-argument, too-many-locals
        original = gorilla.get_original_attribute(keras_sklearn_external.KerasClassifier, 'fit')
        function_info = FunctionInfo('tensorflow.python.keras.wrappers.scikit_learn', 'KerasClassifier')

        # Train data
        input_info_train_data = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno,
                                               function_info, self.mlinspect_optional_code_reference,
                                               self.mlinspect_optional_source_code)
        train_data_op_id = singleton.get_next_op_id()
        operator_context = OperatorContext(OperatorType.TRAIN_DATA, function_info)
        train_data_dag_node = DagNode(train_data_op_id,
                                      BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                                      operator_context,
                                      DagNodeDetails("Train Data", ["array"]),
                                      get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                                     self.mlinspect_optional_source_code))
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
        train_label_op_id = singleton.get_next_op_id()
        train_labels_dag_node = DagNode(train_label_op_id,
                                        BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                                        operator_context,
                                        DagNodeDetails("Train Labels", ["array"]),
                                        get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                                       self.mlinspect_optional_source_code))
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

        dag_node = DagNode(singleton.get_next_op_id(),
                           BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails("Neural Network", []),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code))
        add_dag_node(dag_node, [train_data_dag_node, train_labels_dag_node], estimator_backend_result)
        return self
