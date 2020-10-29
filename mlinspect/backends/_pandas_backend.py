"""
The pandas backend
"""
import os
from collections import namedtuple

import networkx
from pandas import DataFrame, Series
from pandas.core.groupby import DataFrameGroupBy
import numpy

from ._backend import Backend
from ._backend_utils import build_annotation_df_from_iters, \
    create_wrapper_with_annotations
from ._iter_creation import iter_input_data_source, iter_input_annotation_output_resampled, \
    iter_input_annotation_output_map, iter_input_annotation_output_join
from ._pandas_backend_frame_wrapper import MlinspectDataFrame, MlinspectSeries
from ._pandas_wir_processor import PandasWirProcessor
from ..inspections._inspection_input import OperatorContext
from ..instrumentation._dag_node import OperatorType, DagNodeIdentifier


class PandasBackend(Backend):
    """
    The pandas backend
    """

    operator_map = {
        ('pandas.io.parsers', 'read_csv'): OperatorType.DATA_SOURCE,
        ('pandas.core.frame', 'DataFrame'): OperatorType.DATA_SOURCE,
        ('pandas.core.frame', 'dropna'): OperatorType.SELECTION,
        ('pandas.core.frame', '__getitem__'): OperatorType.PROJECTION,  # FIXME: Remove later
        ('pandas.core.frame', '__getitem__', 'Projection'): OperatorType.PROJECTION,
        ('pandas.core.frame', '__getitem__', 'Selection'): OperatorType.SELECTION,
        ('pandas.core.frame', '__setitem__'): OperatorType.PROJECTION_MODIFY,
        ('pandas.core.frame', 'replace'): OperatorType.PROJECTION_MODIFY,
        ('pandas.core.frame', 'merge'): OperatorType.JOIN,
        ('pandas.core.groupby.generic', 'agg'): OperatorType.GROUP_BY_AGG
    }

    replacement_type_map = {
        'mlinspect.backends._pandas_backend_frame_wrapper': 'pandas.core.frame'
    }

    def process_dag(self, dag: networkx.DiGraph) -> networkx.DiGraph:
        """
        Nothing to do here
        """
        return dag

    def __init__(self):
        super().__init__()
        self.input_data = []
        self.df_arg = None
        self.set_key_info = None
        self.select = False
        self.code_reference_to_set_item_op = {}

    def is_responsible_for_call(self, function_info, function_prefix, value=None):
        """Checks whether the backend is responsible for the current method call"""
        function_info = self.replace_wrapper_modules(function_info)
        return function_info in self.operator_map or function_prefix == "pandas"

    def process_wir(self, wir: networkx.DiGraph) -> networkx.DiGraph:
        """
        Special handling to differentiate projections and selections
        """
        PandasWirProcessor().process_wir(wir, self.code_reference_to_set_item_op)
        return wir

    def before_call_used_value(self, function_info, subscript, call_code, value_code, value_value,
                               code_reference):
        """The value or module a function may be called on"""
        # pylint: disable=too-many-arguments
        function_info = self.replace_wrapper_modules(function_info)
        if function_info == ('pandas.core.frame', 'dropna'):
            assert isinstance(value_value, MlinspectDataFrame)
            value_value['mlinspect_index'] = range(0, len(value_value))
        elif function_info == ('pandas.core.frame', '__getitem__'):
            # Can also be a select, but we do only need an index in some cases
            pass
        elif function_info == ('pandas.core.groupby.generic', 'agg'):
            description = value_value.name
            self.code_reference_to_description[code_reference] = description
        elif function_info == ('pandas.core.frame', 'merge'):
            assert isinstance(value_value, MlinspectDataFrame)
            value_value['mlinspect_index_x'] = range(0, len(value_value))
        self.input_data.append(value_value)

    def before_call_used_args(self, function_info, subscript, call_code, args_code, code_reference, store, args_values):
        """The arguments a function may be called with"""
        # pylint: disable=too-many-arguments
        function_info = self.replace_wrapper_modules(function_info)
        if store:
            self.code_reference_to_module[code_reference] = function_info

        if function_info == ('pandas.core.frame', 'merge'):
            assert isinstance(args_values[0], MlinspectDataFrame)
            args_values[0]['mlinspect_index_y'] = range(0, len(args_values[0]))
            self.df_arg = args_values[0]
        elif function_info == ('pandas.core.frame', '__getitem__') and isinstance(args_values, MlinspectSeries):
            self.select = True
            assert isinstance(self.input_data[-1], MlinspectDataFrame)
            self.input_data[-1]['mlinspect_index'] = range(0, len(self.input_data[-1]))
        self.before_call_used_args_add_description(args_values, code_reference, function_info, args_code)

    def before_call_used_args_add_description(self, args_values, code_reference, function_info, args_code):
        """Add special descriptions to certain pandas operators"""
        function_info = self.replace_wrapper_modules(function_info)
        description = None
        if function_info == ('pandas.io.parsers', 'read_csv'):
            filename = args_values[0].split(os.path.sep)[-1]
            description = "{}".format(filename)
        elif function_info == ('pandas.core.frame', 'dropna'):
            description = "dropna"
        elif function_info == ('pandas.core.frame', '__getitem__'):
            if isinstance(args_values, MlinspectSeries):
                self.code_reference_to_set_item_op[code_reference] = 'Selection'
                if args_values.name:
                    description = "Select by series (indirectly using '{}')".format(args_values.name)
                else:
                    description = "Select by series"
            elif isinstance(args_values, str):
                self.code_reference_to_set_item_op[code_reference] = 'Projection'
                key_arg = args_values
                description = "to {}".format([key_arg])
            elif isinstance(args_values, list):
                self.code_reference_to_set_item_op[code_reference] = 'Projection'
                description = "to {}".format(args_values)
        elif function_info == ('pandas.core.frame', '__setitem__'):
            key_arg = args_values
            description = "Sets columns {}".format([key_arg])
            SetKeyInfo = namedtuple("SetKeyInfo", ["code_reference", "function_info", "args_code"])
            self.set_key_info = SetKeyInfo(code_reference, function_info, args_code)
        elif function_info == ('pandas.core.frame', 'groupby'):
            description = "Group by {}, ".format(args_values)
        elif function_info == ('pandas.core.frame', 'replace'):
            description = "Replace {} with {}".format(args_values[0], args_values[1])
        if description:
            self.code_reference_to_description[code_reference] = description

    def before_call_used_kwargs(self, function_info, subscript, call_code, kwargs_code, code_reference, kwargs_values):
        """The keyword arguments a function may be called with"""
        # pylint: disable=too-many-arguments, unused-argument, no-self-use, unnecessary-pass
        function_info = self.replace_wrapper_modules(function_info)
        description = None
        if function_info == ('pandas.core.frame', 'merge'):
            on_column = kwargs_values['on']
            description = "on {}".format(on_column)
        elif function_info == ('pandas.core.groupby.generic', 'agg'):
            old_description = self.code_reference_to_description[code_reference]
            new_description = old_description + " Aggregate: {}".format(list(kwargs_values)[0])
            self.code_reference_to_description[code_reference] = new_description
        if description:
            self.code_reference_to_description[code_reference] = description

    def after_call_used(self, function_info, subscript, call_code, return_value, code_reference):
        """The return value of some function"""
        # pylint: disable=too-many-arguments
        function_info = self.replace_wrapper_modules(function_info)
        self.code_reference_to_module[code_reference] = function_info

        if function_info in {('pandas.io.parsers', 'read_csv'), ('pandas.core.frame', 'DataFrame')}:
            operator_context = OperatorContext(OperatorType.DATA_SOURCE, function_info)
            return_value = execute_inspection_visits_data_source(self, operator_context, code_reference,
                                                                 return_value)
        if function_info == ('pandas.core.groupby.generic', 'agg'):
            operator_context = OperatorContext(OperatorType.GROUP_BY_AGG, function_info)
            return_value = execute_inspection_visits_data_source(self, operator_context, code_reference,
                                                                 return_value.reset_index())
        elif function_info == ('pandas.core.frame', 'dropna'):
            operator_context = OperatorContext(OperatorType.SELECTION, function_info)
            return_value = execute_inspection_visits_unary_operator(self, operator_context, code_reference,
                                                                    self.input_data[-1],
                                                                    self.input_data[-1].annotations,
                                                                    return_value,
                                                                    True)
            self.input_data[-1].drop("mlinspect_index", axis=1, inplace=True)
        elif function_info == ('pandas.core.frame', '__getitem__'):
            if self.select:
                self.select = False
                # Gets converted to Selection later?
                operator_context = OperatorContext(OperatorType.SELECTION, function_info)
                return_value = execute_inspection_visits_unary_operator(self, operator_context, code_reference,
                                                                        self.input_data[-1],
                                                                        self.input_data[-1].annotations,
                                                                        return_value,
                                                                        True)
                self.input_data[-1].drop("mlinspect_index", axis=1, inplace=True)
            elif isinstance(return_value, MlinspectDataFrame):
                operator_context = OperatorContext(OperatorType.PROJECTION, function_info)
                return_value = execute_inspection_visits_unary_operator(self, operator_context, code_reference,
                                                                        self.input_data[-1],
                                                                        self.input_data[-1].annotations,
                                                                        return_value,
                                                                        False)
            elif isinstance(return_value, MlinspectSeries):
                operator_context = OperatorContext(OperatorType.PROJECTION, function_info)
                return_value = execute_inspection_visits_unary_operator(self, operator_context, code_reference,
                                                                        self.input_data[-1],
                                                                        self.input_data[-1].annotations,
                                                                        return_value,
                                                                        False)
        elif function_info == ('pandas.core.frame', 'groupby'):
            description = self.code_reference_to_description[code_reference]
            return_value.name = description  # TODO: Do not use name here but something else to transport the value
        elif function_info == ('pandas.core.frame', 'merge'):
            operator_context = OperatorContext(OperatorType.JOIN, function_info)
            return_value = execute_inspection_visits_join(self, operator_context, code_reference,
                                                          self.input_data[-1],
                                                          self.input_data[-1].annotations,
                                                          self.df_arg,
                                                          self.df_arg.annotations,
                                                          return_value)
            self.input_data[-1].drop("mlinspect_index_x", axis=1, inplace=True)
            self.df_arg.drop("mlinspect_index_y", axis=1, inplace=True)
        elif function_info == ('pandas.core.frame', 'replace'):
            operator_context = OperatorContext(OperatorType.PROJECTION_MODIFY, function_info)
            return_value = execute_inspection_visits_unary_operator(self, operator_context, code_reference,
                                                                    self.input_data[-1],
                                                                    self.input_data[-1].annotations,
                                                                    return_value,
                                                                    False)

        self.input_data.pop()

        return return_value

    def after_call_used_setkey(self, args_code, value_before, value_after):
        """The value before and after some __setkey__ call"""
        # pylint: disable=unused-argument
        code_reference, function_info, args_code = self.set_key_info
        operator_context = OperatorContext(OperatorType.PROJECTION_MODIFY, function_info)
        execute_inspection_visits_unary_operator(self, operator_context, code_reference,
                                                 value_before, value_before.annotations,
                                                 value_after, False)

    def replace_wrapper_modules(self, function_info):
        """Replace the module of mlinspect wrappers with the original modules"""
        if function_info[0] in self.replacement_type_map:
            new_type = self.replacement_type_map[function_info[0]]
            function_info = (new_type, function_info[1])
        return function_info


# -------------------------------------------------------
# Execute inspections functions
# -------------------------------------------------------

def execute_inspection_visits_data_source(backend, operator_context, code_reference, return_value):
    """Execute inspections when the current operator is a data source and does not have parents in the DAG"""
    # pylint: disable=unused-argument
    inspection_count = len(backend.inspections)
    iterators_for_inspections = iter_input_data_source(inspection_count, return_value, operator_context)
    return_value = execute_visits_and_store_results(backend, code_reference, iterators_for_inspections,
                                                    operator_context, return_value)
    return return_value


def execute_inspection_visits_unary_operator(backend, operator_context, code_reference, input_data,
                                             input_annotations, return_value_df, resampled):
    """Execute inspections when the current operator has one parent in the DAG"""
    # pylint: disable=too-many-arguments, unused-argument
    assert not resampled or "mlinspect_index" in return_value_df.columns
    assert isinstance(input_data, (MlinspectDataFrame, MlinspectSeries))
    inspection_count = len(backend.inspections)
    if resampled:
        iterators_for_inspections = iter_input_annotation_output_resampled(inspection_count,
                                                                           input_data,
                                                                           input_annotations,
                                                                           return_value_df,
                                                                           operator_context)
    else:
        iterators_for_inspections = iter_input_annotation_output_map(inspection_count,
                                                                     input_data,
                                                                     input_annotations,
                                                                     return_value_df,
                                                                     operator_context)
    return_value = execute_visits_and_store_results(backend, code_reference, iterators_for_inspections,
                                                    operator_context, return_value_df)
    return return_value


def execute_inspection_visits_join(backend, operator_context, code_reference, input_data_one,
                                   input_annotations_one, input_data_two, input_annotations_two,
                                   return_value_df):
    """Execute inspections when the current operator has one parent in the DAG"""
    # pylint: disable=too-many-arguments, too-many-locals
    assert "mlinspect_index_x" in return_value_df
    assert "mlinspect_index_y" in return_value_df
    assert isinstance(input_data_one, MlinspectDataFrame)
    assert isinstance(input_data_two, MlinspectDataFrame)
    inspection_count = len(backend.inspections)
    iterators_for_inspections = iter_input_annotation_output_join(inspection_count,
                                                                  input_data_one,
                                                                  input_annotations_one,
                                                                  input_data_two,
                                                                  input_annotations_two,
                                                                  return_value_df,
                                                                  operator_context)
    return_value = execute_visits_and_store_results(backend, code_reference, iterators_for_inspections,
                                                    operator_context, return_value_df)
    return return_value


def execute_visits_and_store_results(backend, code_reference, iterators_for_inspections,
                                     operator_context, return_value):
    """
    After creating the iterators we need depending on the operator type, we need to execute the
    generic inspection visits and store the annotations in the resulting data frame
    """
    # pylint: disable=too-many-arguments
    annotation_iterators = []
    for inspection_index, inspection in enumerate(backend.inspections):
        iterator_for_inspection = iterators_for_inspections[inspection_index]
        annotation_iterator = inspection.visit_operator(iterator_for_inspection)
        annotation_iterators.append(annotation_iterator)
    return_value = store_inspection_outputs(backend, annotation_iterators, code_reference, return_value,
                                            operator_context)
    return return_value


# -------------------------------------------------------
# Store inspection results functions
# -------------------------------------------------------

def store_inspection_outputs(backend, annotation_iterators, code_reference, return_value, operator_context):
    """
    Stores the inspection annotations for the rows in the dataframe and the
    inspection annotations for the DAG operators in a map
    """
    dag_node_identifier = DagNodeIdentifier(operator_context.operator, code_reference,
                                            backend.code_reference_to_description.get(code_reference))
    annotations_df = build_annotation_df_from_iters(backend.inspections, annotation_iterators)
    inspection_outputs = {}
    for inspection in backend.inspections:
        inspection_outputs[inspection] = inspection.get_operator_annotation_after_visit()
    backend.dag_node_identifier_to_inspection_output[dag_node_identifier] = inspection_outputs
    new_return_value = create_wrapper_with_annotations(annotations_df, return_value, backend)
    if isinstance(return_value, DataFrame):
        backend.dag_node_identifier_to_columns[dag_node_identifier] = list(new_return_value.columns.values)
    elif isinstance(return_value, Series):
        backend.dag_node_identifier_to_columns[dag_node_identifier] = [new_return_value.name]
    elif isinstance(return_value, DataFrameGroupBy):
        backend.dag_node_identifier_to_columns[dag_node_identifier] = None
    elif isinstance(return_value, numpy.ndarray):
        backend.dag_node_identifier_to_columns[dag_node_identifier] = ["array"]
    else:
        assert False
    return new_return_value
