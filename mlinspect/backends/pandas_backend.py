"""
The pandas backend
"""
import itertools
import os
from collections import namedtuple

import networkx
import pandas

from .backend import Backend
from .backend_utils import get_df_row_iterator, build_annotation_df_from_iters, \
    get_iterator_for_type, create_wrapper_with_annotations, get_annotation_rows
from .pandas_backend_frame_wrapper import MlinspectDataFrame, MlinspectSeries
from .pandas_wir_preprocessor import PandasWirPreprocessor
from ..inspections.inspection_input import InspectionInputUnaryOperator, \
    InspectionInputDataSource, OperatorContext, InspectionInputNAryOperator
from ..instrumentation.dag_node import OperatorType, DagNodeIdentifier


class PandasBackend(Backend):
    """
    The pandas backend
    """

    operator_map = {
        ('pandas.io.parsers', 'read_csv'): OperatorType.DATA_SOURCE,
        ('pandas.core.frame', 'dropna'): OperatorType.SELECTION,
        ('pandas.core.frame', '__getitem__'): OperatorType.PROJECTION,  # FIXME: Remove later
        ('pandas.core.frame', '__getitem__', 'Projection'): OperatorType.PROJECTION,
        ('pandas.core.frame', '__getitem__', 'Selection'): OperatorType.SELECTION,
        ('pandas.core.frame', '__setitem__'): OperatorType.PROJECTION_MODIFY,
        ('pandas.core.frame', 'merge'): OperatorType.JOIN,
        ('pandas.core.groupby.generic', 'agg'): OperatorType.GROUP_BY_AGG
    }

    replacement_type_map = {
        'mlinspect.backends.pandas_backend_frame_wrapper': 'pandas.core.frame'
    }

    def postprocess_dag(self, dag: networkx.DiGraph) -> networkx.DiGraph:
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

    def preprocess_wir(self, wir: networkx.DiGraph) -> networkx.DiGraph:
        """
        Special handling to differentiate projections and selections
        """
        PandasWirPreprocessor().preprocess_wir(wir, self.code_reference_to_set_item_op)
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
            description = "{}".format(filename)  # TODO: Add loaded columns as well
        elif function_info == ('pandas.core.frame', 'dropna'):
            description = "dropna"
        elif function_info == ('pandas.core.frame', '__getitem__'):
            if isinstance(args_values, MlinspectSeries):
                self.code_reference_to_set_item_op[code_reference] = 'Selection'
                description = "Select by series"  # TODO: prettier representation
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
            self.code_reference_to_description[code_reference] = description
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

        if function_info == ('pandas.io.parsers', 'read_csv'):
            operator_context = OperatorContext(OperatorType.DATA_SOURCE, function_info)
            return_value = execute_inspection_visits_no_parents(self, operator_context, code_reference,
                                                                return_value)
        if function_info == ('pandas.core.groupby.generic', 'agg'):
            operator_context = OperatorContext(OperatorType.GROUP_BY_AGG, function_info)
            return_value = execute_inspection_visits_no_parents(self, operator_context, code_reference,
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
        if function_info == ('pandas.core.frame', 'merge'):
            operator_context = OperatorContext(OperatorType.JOIN, function_info)
            return_value = execute_inspection_visits_join(self, operator_context, code_reference,
                                                          self.input_data[-1],
                                                          self.input_data[-1].annotations,
                                                          self.df_arg,
                                                          self.df_arg.annotations,
                                                          return_value)
            self.input_data[-1].drop("mlinspect_index_x", axis=1, inplace=True)
            self.df_arg.drop("mlinspect_index_y", axis=1, inplace=True)

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

def execute_inspection_visits_no_parents(backend, operator_context, code_reference, return_value):
    """Execute inspections when the current operator is a data source and does not have parents in the DAG"""
    # pylint: disable=unused-argument
    inspection_count = len(backend.inspections)
    iterators_for_inspections = iter_input_data_source(inspection_count, return_value)
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
                                                                           return_value_df)
    else:
        iterators_for_inspections = iter_input_annotation_output_df_projection(inspection_count,
                                                                               input_data,
                                                                               input_annotations,
                                                                               return_value_df)
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
    iterators_for_inspections = iter_input_annotation_output_df_pair_df(inspection_count,
                                                                        input_data_one,
                                                                        input_annotations_one,
                                                                        input_data_two,
                                                                        input_annotations_two,
                                                                        return_value_df)
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
        annotation_iterator = inspection.visit_operator(operator_context, iterator_for_inspection)
        annotation_iterators.append(annotation_iterator)
    return_value = store_inspection_outputs(backend, annotation_iterators, code_reference, return_value,
                                            operator_context)
    return return_value


# -------------------------------------------------------
# Functions to create the iterators for the inspections
# -------------------------------------------------------

def iter_input_data_source(inspection_count, output):
    """
    Create an efficient iterator for the inspection input for operators with no parent: Data Source
    """
    output_rows = get_df_row_iterator(output)
    duplicated_output_iterators = itertools.tee(output_rows, inspection_count)
    inspection_iterators = []
    for inspection_index in range(inspection_count):
        output_iterator = duplicated_output_iterators[inspection_index]
        inspection_iterator = map(InspectionInputDataSource, output_iterator)
        inspection_iterators.append(inspection_iterator)

    return inspection_iterators


def iter_input_annotation_output_df_projection(inspection_count, input_data, input_annotations, output):
    """
    Create an efficient iterator for the inspection input for operators with one parent.
    """
    # pylint: disable=too-many-locals
    input_rows = get_iterator_for_type(input_data, True)
    output_rows = get_iterator_for_type(output, False)
    duplicated_input_iterators = itertools.tee(input_rows, inspection_count)
    duplicated_output_iterators = itertools.tee(output_rows, inspection_count)

    inspection_iterators = []
    for inspection_index in range(inspection_count):
        input_iterator = duplicated_input_iterators[inspection_index]
        output_iterator = duplicated_output_iterators[inspection_index]
        annotation_rows = get_annotation_rows(input_annotations, inspection_index)
        inspection_iterator = map(lambda input_tuple: InspectionInputUnaryOperator(*input_tuple),
                                  zip(input_iterator, annotation_rows, output_iterator))
        inspection_iterators.append(inspection_iterator)

    return inspection_iterators


def iter_input_annotation_output_resampled(inspection_count, input_data, input_annotations, output):
    """
    Create an efficient iterator for the inspection input for operators with one parent.
    """
    # pylint: disable=too-many-locals, too-many-arguments
    data_before_with_annotations = pandas.merge(input_data, input_annotations, left_on="mlinspect_index",
                                                right_index=True)
    joined_df = pandas.merge(data_before_with_annotations, output, left_on="mlinspect_index",
                             right_on="mlinspect_index")

    column_index_input_end = len(input_data.columns)
    input_df_view = joined_df.iloc[:, 0:column_index_input_end - 1]
    input_df_view.columns = input_data.columns[0:-1]
    input_rows = get_df_row_iterator(input_df_view)
    duplicated_input_iterators = itertools.tee(input_rows, inspection_count)

    column_index_annotation_end = column_index_input_end + inspection_count
    output_df_view = joined_df.iloc[:, column_index_annotation_end:]
    output_df_view.columns = output.columns[0:-1]
    output_rows = get_df_row_iterator(output_df_view)
    duplicated_output_iterators = itertools.tee(output_rows, inspection_count)

    inspection_iterators = []
    for inspection_index in range(inspection_count):
        input_iterator = duplicated_input_iterators[inspection_index]
        output_iterator = duplicated_output_iterators[inspection_index]
        column_annotation_current_inspection = column_index_input_end + inspection_index
        annotation_rows = get_annotation_rows(joined_df, column_annotation_current_inspection)

        inspection_iterator = map(lambda input_tuple: InspectionInputUnaryOperator(*input_tuple),
                                  zip(input_iterator, annotation_rows, output_iterator))
        inspection_iterators.append(inspection_iterator)

    return inspection_iterators


def iter_input_annotation_output_df_pair_df(inspection_count, x_data, x_annotations, y_data,
                                            y_annotations, output):
    """
    Create an efficient iterator for the inspection input for operators with one parent.
    """
    # pylint: disable=too-many-locals, too-many-arguments
    x_before_with_annotations = pandas.merge(x_data, x_annotations, left_on="mlinspect_index_x",
                                             right_index=True, suffixes=["_x_data", "_x_annot"])
    y_before_with_annotations = pandas.merge(y_data, y_annotations, left_on="mlinspect_index_y",
                                             right_index=True, suffixes=["_y_data", "_y_annot"])
    df_x_output = pandas.merge(x_before_with_annotations, output, left_on="mlinspect_index_x",
                               right_on="mlinspect_index_x", suffixes=["_x", "_output"])
    df_x_output_y = pandas.merge(df_x_output, y_before_with_annotations, left_on="mlinspect_index_y",
                                 right_on="mlinspect_index_y", suffixes=["_x_output", "_y_output"])

    column_index_x_end = len(x_data.columns)

    column_index_output_start = column_index_x_end + inspection_count
    column_index_y_start = column_index_output_start + len(output.columns) - 2
    column_index_y_end = column_index_y_start + len(y_data.columns) - 1

    df_x_output_y = df_x_output_y.drop('mlinspect_index_y', axis=1)

    input_x_view = df_x_output_y.iloc[:, 0:column_index_x_end - 1]
    input_x_view.columns = x_data.columns[0:-1]
    input_y_view = df_x_output_y.iloc[:, column_index_y_start:column_index_y_end]
    input_y_view.columns = y_data.columns[0:-1]
    input_iterators = [get_df_row_iterator(input_x_view), get_df_row_iterator(input_y_view)]
    input_rows = map(list, zip(*input_iterators))
    duplicated_input_iterators = itertools.tee(input_rows, inspection_count)

    output_df_view = df_x_output_y.iloc[:, column_index_output_start:column_index_y_start]
    output_df_view.columns = [column for column in output.columns if
                              (column not in ("mlinspect_index_x", "mlinspect_index_y"))]
    output_rows = get_df_row_iterator(output_df_view)
    duplicated_output_iterators = itertools.tee(output_rows, inspection_count)

    inspection_iterators = []
    for inspection_index in range(inspection_count):
        input_iterator = duplicated_input_iterators[inspection_index]
        output_iterator = duplicated_output_iterators[inspection_index]

        column_annotation_y_current_inspection = column_index_y_end + inspection_index
        column_annotation_x_current_inspection = column_index_x_end + inspection_index
        annotation_iterators = [get_annotation_rows(df_x_output_y, column_annotation_x_current_inspection),
                                get_annotation_rows(df_x_output_y, column_annotation_y_current_inspection)]

        annotation_rows = map(list, zip(*annotation_iterators))

        inspection_iterator = map(lambda input_tuple: InspectionInputNAryOperator(*input_tuple),
                                  zip(input_iterator, annotation_rows, output_iterator))
        inspection_iterators.append(inspection_iterator)

    return inspection_iterators


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
    return new_return_value
