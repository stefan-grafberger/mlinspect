"""
Functions for the implementation for the monkey patched functions
"""
import ast
import dataclasses
import sys
from typing import List

import numpy
from pandas import DataFrame, Series
from scipy.sparse import csr_matrix

from mlinspect.backends._backend import AnnotatedDfObject, BackendResult
from mlinspect.backends._pandas_backend import execute_inspection_visits_data_source
from mlinspect.inspections._inspection_input import OperatorContext
from mlinspect.instrumentation._dag_node import DagNode, OperatorType, CodeReference
from mlinspect.instrumentation._pipeline_executor import singleton


@dataclasses.dataclass(frozen=True)
class InputInfo:
    """ WIP experiments """
    dag_node: DagNode
    annotated_dfobject: AnnotatedDfObject


def execute_patched_func(original_func, execute_inspections_func, *args, **kwargs):
    """
    Detects whether the function call comes directly from user code and decides whether to execute the original
    function or the patched variant.
    """
    # Performance aspects: https://gist.github.com/JettJones/c236494013f22723c1822126df944b12
    # We have to do this max once unnecessarily per user function call as long as we always set the singleton
    #  lineno to -1 at the end of every patched function
    # We will have to think about indirect calls. E.g. pandas functions use other pandas functions internally
    #  For pandas, maybe we only want to consider pandas functions directly called by the user
    #  For sklearn, we might want to consider all indirect fit/transform calls

    # CPython implementation detail: This function should be used for internal and specialized purposes only.
    #  It is not guaranteed to exist in all implementations of Python.
    #  inspect.getcurrentframe() also only does return `sys._getframe(1) if hasattr(sys, "_getframe") else None`
    #  We can execute one hasattr check right at the beginning of the mlinspect execution

    # if singleton.track_code_references and singleton.lineno_next_call_or_subscript == -1:
    #    original_func(self_obj, *args, **kwargs)
    #    return

    caller_filename = sys._getframe(2).f_code.co_filename  # pylint: disable=protected-access

    if caller_filename != singleton.source_code_path:
        result = original_func(*args, **kwargs)
    elif singleton.track_code_references:
        call_ast_node = ast.Call(lineno=singleton.lineno_next_call_or_subscript,
                                 col_offset=singleton.col_offset_next_call_or_subscript,
                                 end_lineno=singleton.end_lineno_next_call_or_subscript,
                                 end_col_offset=singleton.end_col_offset_next_call_or_subscript)
        caller_source_code = ast.get_source_segment(singleton.source_code, node=call_ast_node)
        caller_lineno = singleton.lineno_next_call_or_subscript
        op_id = singleton.get_next_op_id()
        caller_code_reference = CodeReference(singleton.lineno_next_call_or_subscript,
                                              singleton.col_offset_next_call_or_subscript,
                                              singleton.end_lineno_next_call_or_subscript,
                                              singleton.end_col_offset_next_call_or_subscript)
        result = execute_inspections_func(op_id, caller_filename, caller_lineno, caller_code_reference,
                                          caller_source_code)
        # singleton.lineno_next_call_or_subscript = -1
    else:
        op_id = singleton.get_next_op_id()
        caller_lineno = sys._getframe(2).f_lineno  # pylint: disable=protected-access
        result = execute_inspections_func(op_id, caller_filename, caller_lineno, None, None)
        # singleton.lineno_next_call_or_subscript = -1
    return result


def execute_patched_func_indirect_allowed(execute_inspections_func):
    """
    Detects whether the function call comes directly from user code and decides whether to execute the original
    function or the patched variant.
    """
    # Performance aspects: https://gist.github.com/JettJones/c236494013f22723c1822126df944b12
    # We have to do this max once unnecessarily per user function call as long as we always set the singleton
    #  lineno to -1 at the end of every patched function
    # We will have to think about indirect calls. E.g. pandas functions use other pandas functions internally
    #  For pandas, maybe we only want to consider pandas functions directly called by the user
    #  For sklearn, we might want to consider all indirect fit/transform calls

    # CPython implementation detail: This function should be used for internal and specialized purposes only.
    #  It is not guaranteed to exist in all implementations of Python.
    #  inspect.getcurrentframe() also only does return `sys._getframe(1) if hasattr(sys, "_getframe") else None`
    #  We can execute one hasattr check right at the beginning of the mlinspect execution

    # if singleton.track_code_references and singleton.lineno_next_call_or_subscript == -1:
    #    original_func(self_obj, *args, **kwargs)
    #    return

    frame = sys._getframe(2)  # pylint: disable=protected-access

    while frame.f_code.co_filename != singleton.source_code_path:
        frame = frame.f_back

    caller_filename = frame.f_code.co_filename
    caller_lineno = frame.f_lineno

    if singleton.track_code_references:
        call_ast_node = ast.Call(lineno=singleton.lineno_next_call_or_subscript,
                                 col_offset=singleton.col_offset_next_call_or_subscript,
                                 end_lineno=singleton.end_lineno_next_call_or_subscript,
                                 end_col_offset=singleton.end_col_offset_next_call_or_subscript)
        caller_source_code = ast.get_source_segment(singleton.source_code, node=call_ast_node)
        caller_lineno = singleton.lineno_next_call_or_subscript
        op_id = singleton.get_next_op_id()
        caller_code_reference = CodeReference(singleton.lineno_next_call_or_subscript,
                                               singleton.col_offset_next_call_or_subscript,
                                               singleton.end_lineno_next_call_or_subscript,
                                               singleton.end_col_offset_next_call_or_subscript)
        result = execute_inspections_func(op_id, caller_filename, caller_lineno, caller_code_reference,
                                          caller_source_code)
        # singleton.lineno_next_call_or_subscript = -1
    else:
        op_id = singleton.get_next_op_id()
        caller_lineno = sys._getframe(2).f_lineno  # pylint: disable=protected-access
        result = execute_inspections_func(op_id, caller_filename, caller_lineno, None, None)
        # singleton.lineno_next_call_or_subscript = -1
    return result


def get_input_info(df_object, caller_filename, lineno, function_info, optional_code_reference, optional_source_code) \
        -> InputInfo:
    """
    Uses the patched _mlinspect_dag_node attribute and the singleton.op_id_to_dag_node map to find the parent DAG node
    for the DAG node we want to insert in the next step.
    """
    # pylint: disable=too-many-arguments, unused-argument, protected-access, unused-variable, too-many-locals
    if isinstance(df_object, DataFrame):
        columns = list(df_object.columns)  # TODO: Update this for numpy arrays etc. later
    elif isinstance(df_object, Series):
        columns = [df_object.name]
    elif isinstance(df_object, (csr_matrix, numpy.ndarray)):
        columns = ['array']
    else:
        raise NotImplementedError("TODO: Mlinspect info storage for type: '{}'".format(type(df_object)))
    if hasattr(df_object, "_mlinspect_annotation"):
        input_op_id = df_object._mlinspect_dag_node
        input_dag_node = singleton.op_id_to_dag_node[input_op_id]
        annotation_df = df_object._mlinspect_annotation
        input_info = InputInfo(input_dag_node, AnnotatedDfObject(df_object, annotation_df))
    else:
        operator_context = OperatorContext(OperatorType.DATA_SOURCE, function_info)
        backend_result = execute_inspection_visits_data_source(operator_context, df_object)
        if optional_code_reference:
            code_reference = "({})".format(optional_source_code)
        else:
            code_reference = ""
        description = "Warning! Operator {}:{} {} encountered a DataFrame resulting from an operation " \
                      "without mlinspect support!".format(caller_filename, lineno, code_reference)
        missing_op_id = singleton.get_next_missing_op_id()
        input_dag_node = DagNode(missing_op_id, caller_filename, lineno, OperatorType.DATA_SOURCE,
                                 description=description,
                                 columns=columns, optional_code_reference=optional_code_reference,
                                 optional_source_code=optional_source_code)
        add_dag_node(input_dag_node, [], backend_result)
        annotation_df = backend_result.annotated_dfobject.result_annotation
        input_info = InputInfo(input_dag_node, AnnotatedDfObject(df_object, annotation_df))
    return input_info


def add_dag_node(dag_node: DagNode, dag_node_parents: List[DagNode], backend_result: BackendResult):
    """
    Inserts a new node into the DAG
    """
    # pylint: disable=protected-access
    print("")
    print("{}:{}: {}".format(dag_node.caller_filename, dag_node.lineno, dag_node.module))

    print("source code: {}".format(dag_node.optional_source_code))
    annotated_df = backend_result.annotated_dfobject

    if annotated_df.result_data is not None:
        annotated_df.result_data._mlinspect_dag_node = dag_node.node_id
        if annotated_df.result_annotation is not None:
            # TODO: Remove this branching once we support all operators with DAG node mapping
            annotated_df.result_data._mlinspect_annotation = annotated_df.result_annotation
    if dag_node_parents:
        for parent in dag_node_parents:
            singleton.inspection_results.dag.add_edge(parent, dag_node)
    else:
        singleton.inspection_results.dag.add_node(dag_node)
    singleton.op_id_to_dag_node[dag_node.node_id] = dag_node
    if annotated_df is not None:
        singleton.inspection_results.dag_node_to_inspection_results[dag_node] = backend_result.dag_node_annotation
