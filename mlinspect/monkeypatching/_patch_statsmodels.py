"""
Monkey patching for numpy
"""
import gorilla
from statsmodels import api
from statsmodels.api import datasets

from mlinspect import DagNode, BasicCodeLocation, DagNodeDetails
from mlinspect.backends._pandas_backend import PandasBackend
from mlinspect.backends._sklearn_backend import SklearnBackend
from mlinspect.inspections._inspection_input import OperatorContext, FunctionInfo, OperatorType
from mlinspect.instrumentation import _pipeline_executor
from mlinspect.instrumentation._pipeline_executor import singleton
from mlinspect.monkeypatching._monkey_patching_utils import execute_patched_func, add_dag_node, \
    get_optional_code_info_or_none, get_input_info, execute_patched_func_no_op_id


@gorilla.patches(api)
class StatsmodelApiPatching:
    """ Patches for sklearn """

    # pylint: disable=too-few-public-methods

    @gorilla.name('add_constant')
    @gorilla.settings(allow_hit=True)
    def patched_random(*args, **kwargs):
        """ Patch for ('statsmodel.api', 'add_constant') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(api, 'add_constant')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = FunctionInfo('statsmodel.api', 'add_constant')
            input_info = get_input_info(args[0], caller_filename, lineno, function_info, optional_code_reference,
                                        optional_source_code)

            operator_context = OperatorContext(OperatorType.PROJECTION_MODIFY, function_info)
            input_infos = SklearnBackend.before_call(operator_context, [input_info.annotated_dfobject])
            result = original(input_infos[0].result_data, *args[1:], **kwargs)
            backend_result = SklearnBackend.after_call(operator_context,
                                                       input_infos,
                                                       result)
            new_return_value = backend_result.annotated_dfobject.result_data

            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails("Adds const column", ["array"]),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            add_dag_node(dag_node, [input_info.dag_node], backend_result)

            return new_return_value
        return execute_patched_func(original, execute_inspections, *args, **kwargs)


@gorilla.patches(datasets)
class StatsmodelsDatasetPatching:
    """ Patches for pandas """

    # pylint: disable=too-few-public-methods

    @gorilla.name('get_rdataset')
    @gorilla.settings(allow_hit=True)
    def patched_read_csv(*args, **kwargs):
        """ Patch for ('statsmodels.datasets', 'get_rdataset') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(datasets, 'get_rdataset')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = FunctionInfo('statsmodels.datasets', 'get_rdataset')

            operator_context = OperatorContext(OperatorType.DATA_SOURCE, function_info)
            input_infos = PandasBackend.before_call(operator_context, [])
            result = original(*args, **kwargs)
            backend_result = PandasBackend.after_call(operator_context,
                                                      input_infos,
                                                      result.data)
            result.data = backend_result.annotated_dfobject.result_data
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(result.title, list(result.data.columns)),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            add_dag_node(dag_node, [], backend_result)
            return result

        return execute_patched_func(original, execute_inspections, *args, **kwargs)


@gorilla.patches(api.OLS)
class StatsmodelsOlsPatching:
    """ Patches for statsmodel OLS"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *args, **kwargs):
        """ Patch for ('statsmodel.api', 'OLS') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init, too-many-locals
        original = gorilla.get_original_attribute(api.OLS, '__init__')

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, *args, **kwargs)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, *args, **kwargs)

    @gorilla.name('fit')
    @gorilla.settings(allow_hit=True)
    def patched_fit(self, *args, **kwargs):
        """ Patch for ('statsmodel.api.OLS', 'fit') """
        # pylint: disable=no-method-argument, too-many-locals
        original = gorilla.get_original_attribute(api.OLS, 'fit')
        function_info = FunctionInfo('statsmodel.api.OLS', 'fit')

        # Train data
        input_info_train_data = get_input_info(self.data.exog,  # pylint: disable=no-member
                                               self.mlinspect_caller_filename, self.mlinspect_lineno,
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
                                                        self.data.exog)  # pylint: disable=no-member
        add_dag_node(train_data_dag_node, [input_info_train_data.dag_node], data_backend_result)

        # Train labels
        operator_context = OperatorContext(OperatorType.TRAIN_LABELS, function_info)
        input_info_train_labels = get_input_info(self.data.endog,  # pylint: disable=no-member
                                                 self.mlinspect_caller_filename, self.mlinspect_lineno,
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
                                                         self.data.endog)  # pylint: disable=no-member
        add_dag_node(train_labels_dag_node, [input_info_train_labels.dag_node], label_backend_result)

        # Estimator
        operator_context = OperatorContext(OperatorType.ESTIMATOR, function_info)
        input_dfs = [data_backend_result.annotated_dfobject, label_backend_result.annotated_dfobject]
        input_infos = SklearnBackend.before_call(operator_context, input_dfs)
        result = original(self, *args, **kwargs)
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
        return result
