"""
Monkey patching for pandas
"""
import os

import gorilla
import pandas

from mlinspect import OperatorType, DagNode
from mlinspect.backends._pandas_backend import PandasBackend
from mlinspect.inspections._inspection_input import OperatorContext
from mlinspect.monkeypatching.monkey_patching_utils import execute_patched_func, get_input_info, add_dag_node, \
    get_dag_node_for_id, execute_patched_func_no_op_id


@gorilla.patches(pandas)
class PandasPatching:
    """ Patches for pandas """

    # pylint: disable=too-few-public-methods

    @gorilla.name('read_csv')
    @gorilla.settings(allow_hit=True)
    def patched_read_csv(*args, **kwargs):
        """ Patch for ('pandas.io.parsers', 'read_csv') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(pandas, 'read_csv')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = ('pandas.io.parsers', 'read_csv')

            operator_context = OperatorContext(OperatorType.DATA_SOURCE, function_info)
            input_infos = PandasBackend.before_call(operator_context, [])
            result = original(*args, **kwargs)
            backend_result = PandasBackend.after_call(operator_context,
                                                      input_infos,
                                                      result)

            description = "{}".format(args[0].split(os.path.sep)[-1])
            dag_node = DagNode(op_id, caller_filename, lineno, OperatorType.DATA_SOURCE, function_info, description,
                               list(result.columns), optional_code_reference, optional_source_code)
            add_dag_node(dag_node, [], backend_result)
            return result

        return execute_patched_func(original, execute_inspections, *args, **kwargs)


@gorilla.patches(pandas.DataFrame)
class DataFramePatching:
    """ Patches for 'pandas.core.frame' """

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *args, **kwargs):
        """ Patch for ('pandas.core.frame', 'DataFrame') """
        original = gorilla.get_original_attribute(pandas.DataFrame, '__init__')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = ('pandas.core.frame', 'DataFrame')
            operator_context = OperatorContext(OperatorType.DATA_SOURCE, function_info)
            input_infos = PandasBackend.before_call(operator_context, [])
            original(self, *args, **kwargs)
            result = self
            backend_result = PandasBackend.after_call(operator_context, input_infos, result)

            columns = list(self.columns)  # pylint: disable=no-member
            dag_node = DagNode(op_id, caller_filename, lineno, OperatorType.DATA_SOURCE, function_info,
                               "", columns, optional_code_reference, optional_source_code)
            add_dag_node(dag_node, [], backend_result)

        execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('dropna')
    @gorilla.settings(allow_hit=True)
    def patched_dropna(self, *args, **kwargs):
        """ Patch for ('pandas.core.frame', 'dropna') """
        original = gorilla.get_original_attribute(pandas.DataFrame, 'dropna')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = ('pandas.core.frame', 'dropna')

            input_info = get_input_info(self, caller_filename, lineno, function_info, optional_code_reference,
                                        optional_source_code)
            operator_context = OperatorContext(OperatorType.SELECTION, function_info)
            input_infos = PandasBackend.before_call(operator_context, [input_info.annotated_dfobject])
            # No input_infos copy needed because it's only a selection and the rows not being removed don't change
            result = original(input_infos[0].result_data, *args[1:], **kwargs)
            if result is None:
                raise NotImplementedError("TODO: Support inplace dropna")
            backend_result = PandasBackend.after_call(operator_context,
                                                      input_infos,
                                                      result)
            result = backend_result.annotated_dfobject.result_data
            dag_node = DagNode(op_id, caller_filename, lineno, OperatorType.SELECTION, function_info,
                               "dropna", list(result.columns), optional_code_reference, optional_source_code)
            add_dag_node(dag_node, [input_info.dag_node], backend_result)

            return result

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__getitem__')
    @gorilla.settings(allow_hit=True)
    def patched__getitem__(self, *args, **kwargs):
        """ Patch for ('pandas.core.frame', '__getitem__') """
        original = gorilla.get_original_attribute(pandas.DataFrame, '__getitem__')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = ('pandas.core.frame', '__getitem__')
            input_info = get_input_info(self, caller_filename, lineno, function_info, optional_code_reference,
                                        optional_source_code)
            if isinstance(args[0], str):  # Projection to Series
                columns = [args[0]]
                operator_context = OperatorContext(OperatorType.PROJECTION, function_info)
                dag_node = DagNode(op_id, caller_filename, lineno, OperatorType.PROJECTION, function_info,
                                   "to {}".format(columns), columns, optional_code_reference, optional_source_code)
            elif isinstance(args[0], list) and isinstance(args[0][0], str):  # Projection to DF
                columns = args[0]
                operator_context = OperatorContext(OperatorType.PROJECTION, function_info)
                dag_node = DagNode(op_id, caller_filename, lineno, OperatorType.PROJECTION, function_info,
                                   "to {}".format(columns), columns, optional_code_reference, optional_source_code)
            elif isinstance(args[0], pandas.Series):  # Selection
                operator_context = OperatorContext(OperatorType.SELECTION, function_info)
                columns = list(self.columns)  # pylint: disable=no-member
                dag_node = DagNode(op_id, caller_filename, lineno, OperatorType.SELECTION, function_info,
                                   "Select by Series", columns, optional_code_reference, optional_source_code)
            else:
                raise NotImplementedError()
            input_infos = PandasBackend.before_call(operator_context, [input_info.annotated_dfobject])
            result = original(input_infos[0].result_data, *args, **kwargs)
            backend_result = PandasBackend.after_call(operator_context,
                                                      input_infos,
                                                      result)
            result = backend_result.annotated_dfobject.result_data
            add_dag_node(dag_node, [input_info.dag_node], backend_result)

            return result

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__setitem__')
    @gorilla.settings(allow_hit=True)
    def patched__setitem__(self, *args, **kwargs):
        """ Patch for ('pandas.core.frame', '__setitem__') """
        original = gorilla.get_original_attribute(pandas.DataFrame, '__setitem__')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            module = ('pandas.core.frame', '__setitem__')
            input_info = get_input_info(self, caller_filename, lineno, module, optional_code_reference,
                                        optional_source_code)
            input_dag_node = input_info.dag_node
            engine_input = [input_info.engine_input]
            # if isinstance(args[0], str):
            #     user_operation = LogicalProjectAssignColumn(1, args[0], args[1])
            #     fallback = partial(original, self, *args, **kwargs)
            #     engine_result = _pipeline_executor.singleton.engine.run(engine_input, user_operation, fallback)
            #     result = engine_result.user_op_result.to_pandas_df()
            #     columns = list(result.columns)
            #     description = "modifies {}".format([args[0]])
            # else:
            #     raise NotImplementedError("TODO: Handling __setitem__ for key type {}".format(type(args[0])))
            # dag_node = DagNode2(op_id, caller_filename, lineno, OperatorType2.PROJECTION_MODIFY, module,
            #                     description, columns, optional_code_reference, optional_source_code)
            # add_dag_node(dag_node, [input_dag_node], result, engine_result)
            # return result

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('replace')
    @gorilla.settings(allow_hit=True)
    def patched_replace(self, *args, **kwargs):
        """ Patch for ('pandas.core.frame', 'replace') """
        original = gorilla.get_original_attribute(pandas.DataFrame, 'replace')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = ('pandas.core.frame', 'replace')

            input_info = get_input_info(self, caller_filename, lineno, function_info, optional_code_reference,
                                        optional_source_code)
            operator_context = OperatorContext(OperatorType.PROJECTION_MODIFY, function_info)
            input_infos = PandasBackend.before_call(operator_context, [input_info.annotated_dfobject])
            # No input_infos copy needed because it's only a selection and the rows not being removed don't change
            result = original(input_infos[0].result_data, *args, **kwargs)
            backend_result = PandasBackend.after_call(operator_context,
                                                      input_infos,
                                                      result)
            result = backend_result.annotated_dfobject.result_data
            if isinstance(args[0], dict):
                raise NotImplementedError("TODO: Add support for replace with dicts")
            description = "Replace '{}' with '{}'".format(args[0], args[1])
            dag_node = DagNode(op_id, caller_filename, lineno, OperatorType.PROJECTION_MODIFY, function_info,
                               description, list(result.columns), optional_code_reference, optional_source_code)
            add_dag_node(dag_node, [input_info.dag_node], backend_result)

            return result

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('merge')
    @gorilla.settings(allow_hit=True)
    def patched_merge(self, *args, **kwargs):
        """ Patch for ('pandas.core.frame', 'merge') """
        original = gorilla.get_original_attribute(pandas.DataFrame, 'merge')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = ('pandas.core.frame', 'merge')

            input_info_a = get_input_info(self, caller_filename, lineno, function_info, optional_code_reference,
                                          optional_source_code)
            input_info_b = get_input_info(args[0], caller_filename, lineno, function_info, optional_code_reference,
                                          optional_source_code)
            operator_context = OperatorContext(OperatorType.JOIN, function_info)
            input_infos = PandasBackend.before_call(operator_context, [input_info_a.annotated_dfobject,
                                                                       input_info_b.annotated_dfobject])
            # No input_infos copy needed because it's only a selection and the rows not being removed don't change
            result = original(input_infos[0].result_data, input_infos[1].result_data, *args[1:], **kwargs)
            backend_result = PandasBackend.after_call(operator_context,
                                                      input_infos,
                                                      result)
            result = backend_result.annotated_dfobject.result_data
            description = "on '{}'".format(kwargs['on'])
            dag_node = DagNode(op_id, caller_filename, lineno, OperatorType.JOIN, function_info,
                               description, list(result.columns), optional_code_reference, optional_source_code)
            add_dag_node(dag_node, [input_info_a.dag_node, input_info_b.dag_node], backend_result)

            return result

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('merge')
    @gorilla.settings(allow_hit=True)
    def patched_merge(self, *args, **kwargs):
        """ Patch for ('pandas.core.frame', 'merge') """
        original = gorilla.get_original_attribute(pandas.DataFrame, 'merge')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = ('pandas.core.frame', 'merge')

            input_info_a = get_input_info(self, caller_filename, lineno, function_info, optional_code_reference,
                                          optional_source_code)
            input_info_b = get_input_info(args[0], caller_filename, lineno, function_info, optional_code_reference,
                                          optional_source_code)
            operator_context = OperatorContext(OperatorType.JOIN, function_info)
            input_infos = PandasBackend.before_call(operator_context, [input_info_a.annotated_dfobject,
                                                                       input_info_b.annotated_dfobject])
            # No input_infos copy needed because it's only a selection and the rows not being removed don't change
            result = original(input_infos[0].result_data, input_infos[1].result_data, *args[1:], **kwargs)
            backend_result = PandasBackend.after_call(operator_context,
                                                      input_infos,
                                                      result)
            result = backend_result.annotated_dfobject.result_data
            description = "on '{}'".format(kwargs['on'])
            dag_node = DagNode(op_id, caller_filename, lineno, OperatorType.JOIN, function_info,
                               description, list(result.columns), optional_code_reference, optional_source_code)
            add_dag_node(dag_node, [input_info_a.dag_node, input_info_b.dag_node], backend_result)

            return result

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('groupby')
    @gorilla.settings(allow_hit=True)
    def patched_merge(self, *args, **kwargs):
        """ Patch for ('pandas.core.frame', 'groupby') """
        original = gorilla.get_original_attribute(pandas.DataFrame, 'groupby')

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = ('pandas.core.frame', 'groupby')
            # We ignore groupbys, we only do something with aggs

            input_info = get_input_info(self, caller_filename, lineno, function_info, optional_code_reference,
                                        optional_source_code)
            result = original(self, *args, **kwargs)
            result._mlinspect_dag_node = input_info.dag_node.node_id

            return result

        return execute_patched_func_no_op_id(original, execute_inspections, self, *args, **kwargs)


@gorilla.patches(pandas.core.groupby.generic.DataFrameGroupBy)
class DataFrameGroupByPatching:
    """ Patches for 'pandas.core.groupby.generic' """

    @gorilla.name('agg')
    @gorilla.settings(allow_hit=True)
    def patched_agg(self, *args, **kwargs):
        """ Patch for ('pandas.core.groupby.generic', 'agg') """
        original = gorilla.get_original_attribute(pandas.core.groupby.generic.DataFrameGroupBy, 'agg')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = ('pandas.core.groupby.generic', 'agg')
            if not hasattr(self, '_mlinspect_dag_node'):
                raise NotImplementedError("TODO: Support agg if groupby happened in external code")
            input_dag_node = get_dag_node_for_id(self._mlinspect_dag_node)

            operator_context = OperatorContext(OperatorType.GROUP_BY_AGG, function_info)

            input_infos = PandasBackend.before_call(operator_context, [])
            result = original(self, *args, **kwargs)
            backend_result = PandasBackend.after_call(operator_context,
                                                      input_infos,
                                                      result)

            if len(args) > 0:
                description = "Groupby '{}', Aggregate: '{}'".format(result.index.name, args)
            else:
                description = "Groupby '{}', Aggregate: '{}'".format(result.index.name, kwargs)
            columns = [result.index.name] + list(result.columns)
            dag_node = DagNode(op_id, caller_filename, lineno, OperatorType.GROUP_BY_AGG, function_info, description,
                               columns, optional_code_reference, optional_source_code)
            add_dag_node(dag_node, [input_dag_node], backend_result)

            return result

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)


@gorilla.patches(pandas.core.indexing._LocIndexer)  # pylint: disable=protected-access
class LocIndexerPatching:
    """ Patches for 'pandas.core.series' """

    # pylint: disable=too-few-public-methods, too-many-locals

    @gorilla.name('__getitem__')
    @gorilla.settings(allow_hit=True)
    def patched__getitem__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', 'Series') """
        original = gorilla.get_original_attribute(
            pandas.core.indexing._LocIndexer, '__getitem__')  # pylint: disable=protected-access

        # if call_info_singleton.column_transformer_active:
        #     op_id = _pipeline_executor.singleton.get_next_op_id()
        #     caller_filename = call_info_singleton.transformer_filename
        #     lineno = call_info_singleton.transformer_lineno
        #     module = call_info_singleton.module
        #     optional_code_reference = call_info_singleton.transformer_optional_code_reference
        #     optional_source_code = call_info_singleton.transformer_optional_source_code
        #
        #     if isinstance(args[0], tuple) and not args[0][0].start and not args[0][0].stop \
        #             and isinstance(args[0][1], list) and isinstance(args[0][1][0], str):
        #         # Projection to one or multiple columns, return value is df
        #         columns = args[0][1]
        #         user_operation = LogicalProjection(1, columns)
        #         dag_node = DagNode2(op_id, caller_filename, lineno, OperatorType2.PROJECTION, module,
        #                             "to {}".format(columns), columns, optional_code_reference, optional_source_code)
        #     else:
        #         raise NotImplementedError()
        #
        #     input_info = get_input_info(self.obj, caller_filename,  # pylint: disable=no-member
        #                                 lineno, module, optional_code_reference, optional_source_code)
        #     engine_input = [input_info.engine_input]
        #     fallback = partial(original, self, *args, **kwargs)
        #     engine_result = _pipeline_executor.singleton.engine.run(engine_input, user_operation, fallback)
        #
        #     result = engine_result.user_op_result.to_pandas_df()
        #     add_dag_node(dag_node, [input_info.dag_node], result, engine_result)
        #
        #     dag_node = DagNode2(op_id, caller_filename, lineno, OperatorType2.PROJECTION, module,
        #                         "to {}".format(columns), columns, optional_code_reference, optional_source_code)
        #     add_dag_node(dag_node, [input_info.dag_node], result, engine_result)
        # else:
        result = original(self, *args, **kwargs)

        return result


@gorilla.patches(pandas.Series)
class SeriesPatching:
    """ Patches for 'pandas.core.series' """

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', 'Series') """
        original = gorilla.get_original_attribute(pandas.Series, '__init__')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = ('pandas.core.series', 'Series')

            operator_context = OperatorContext(OperatorType.DATA_SOURCE, function_info)
            input_infos = PandasBackend.before_call(operator_context, [])
            original(self, *args, **kwargs)
            result = self
            backend_result = PandasBackend.after_call(operator_context,
                                                      input_infos,
                                                      result)

            if self.name:
                columns = list(self.name)  # pylint: disable=no-member
            else:
                columns = ["_1"]
            dag_node = DagNode(op_id, caller_filename, lineno, OperatorType.DATA_SOURCE, function_info,
                               "", columns, optional_code_reference, optional_source_code)
            add_dag_node(dag_node, [], backend_result)

        execute_patched_func(original, execute_inspections, self, *args, **kwargs)
