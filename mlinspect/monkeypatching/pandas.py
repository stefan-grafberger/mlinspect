"""
Monkey patching for pandas
"""
import os
from functools import partial

import gorilla
import pandas

from mlinspect import OperatorType, DagNode
from mlinspect.backends._backend import AnnotatedDfObject
from mlinspect.backends._pandas_backend import PandasBackend
from mlinspect.inspections._inspection_input import OperatorContext
from mlinspect.instrumentation import _pipeline_executor
from mlinspect.monkeypatching.monkey_patching_utils import execute_patched_func, get_input_info, add_dag_node


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
            input_infos = PandasBackend.before_call(operator_context,
                                                    [])
            original(self, *args, **kwargs)
            result = self
            backend_result = PandasBackend.after_call(operator_context,
                                                      input_infos,
                                                      result)

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
            module = ('pandas.core.frame', 'dropna')

            input_info = get_input_info(self, caller_filename, lineno, module, optional_code_reference,
                                        optional_source_code)
            # TODO: Test passing of optional args/kwargs
            # TODO: Maybe we do not want to have UDFs like that in our engine DAG but introduce corresponding
            #  abstract functions that then map to different implementations in different backends
            # FIXME: Introduce a flag to LogicalSelection to indicate whether e.g. inspection annotations with
            #  null values can change the result of the user function call if we are not careful?
            partial_dropna = lambda x: original(x, *args, **kwargs)
            # user_operation = LogicalSelection(1, partial_dropna, DataframeType.PANDAS_DF)
            engine_input = [input_info.engine_input]
            fallback = partial(original, self, *args, **kwargs)
            # engine_result = _pipeline_executor.singleton.engine.run(engine_input, user_operation, fallback)
            # user_op_result = engine_result.user_op_result.to_pandas_df()
            # dag_node = DagNode2(op_id, caller_filename, lineno, OperatorType2.SELECTION, module,
            #                    "dropna", list(user_op_result.columns), optional_code_reference, optional_source_code)
            # add_dag_node(dag_node, [input_info.dag_node], user_op_result, engine_result)
            # return user_op_result

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__getitem__')
    @gorilla.settings(allow_hit=True)
    def patched__getitem__(self, *args, **kwargs):
        """ Patch for ('pandas.core.frame', '__getitem__') """
        original = gorilla.get_original_attribute(pandas.DataFrame, '__getitem__')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            module = ('pandas.core.frame', '__getitem__')
            input_info = get_input_info(self, caller_filename, lineno, module, optional_code_reference,
                                        optional_source_code)
            # if isinstance(args[0], str):  # Projection to Series
            #     columns = [args[0]]
            #     user_operation = LogicalProjection(1, columns)
            #     dag_node = DagNode2(op_id, caller_filename, lineno, OperatorType2.PROJECTION, module,
            #                         "to {}".format(columns), columns, optional_code_reference, optional_source_code)
            # elif isinstance(args[0], list) and isinstance(args[0][0], str):  # Projection to DF
            #     columns = args[0]
            #     user_operation = LogicalProjection(1, columns)
            #     dag_node = DagNode2(op_id, caller_filename, lineno, OperatorType2.PROJECTION, module,
            #                         "to {}".format(columns), columns, optional_code_reference, optional_source_code)
            # elif isinstance(args[0], pandas.Series):  # Selection
            #     partial_select = lambda x: original(x, args[0])
            #     user_operation = LogicalSelection(1, partial_select, DataframeType.PANDAS_DF)
            #     columns = list(self.columns)  # pylint: disable=no-member
            #     dag_node = DagNode2(op_id, caller_filename, lineno, OperatorType2.SELECTION, module,
            #                         "Select by Series", columns, optional_code_reference, optional_source_code)
            # else:
            #     raise NotImplementedError()
            # engine_input = [input_info.engine_input]
            # fallback = partial(original, self, *args, **kwargs)
            # engine_result = _pipeline_executor.singleton.engine.run(engine_input, user_operation, fallback)
            #
            # if isinstance(args[0], str):
            #     result = engine_result.user_op_result.to_pandas_series()
            # else:
            #     result = engine_result.user_op_result.to_pandas_df()
            # add_dag_node(dag_node, [input_info.dag_node], result, engine_result)
            # return result

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
        #     result = original(self, *args, **kwargs)
        #
        # return result


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
