"""
The pandas backend
"""
import os
from collections import namedtuple

import networkx
import pandas

from ..inspections.inspection_input import InspectionInputUnaryOperator, \
    InspectionInputDataSource, OperatorContext, InspectionInputNAryOperator
from .backend import Backend
from .backend_utils import get_df_row_iterator, build_annotation_df_from_iters, \
    get_series_row_iterator
from .pandas_backend_frame_wrapper import MlinspectDataFrame, MlinspectSeries
from .pandas_wir_preprocessor import PandasWirPreprocessor
from ..instrumentation.dag_node import OperatorType, DagNodeIdentifier


class PandasBackend(Backend):
    """
    The pandas backend
    """

    prefix = "pandas"

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
        if function_info == ('pandas.core.frame', 'dropna'):
            assert isinstance(value_value, MlinspectDataFrame)
            value_value['mlinspect_index'] = range(1, len(value_value) + 1)
        elif function_info == ('pandas.core.frame', '__getitem__'):
            # TODO: Can this also be a select?
            assert isinstance(value_value, MlinspectDataFrame)
            value_value['mlinspect_index'] = range(1, len(value_value) + 1)
        elif function_info == ('pandas.core.groupby.generic', 'agg'):
            description = value_value.name
            self.code_reference_to_description[code_reference] = description
        elif function_info == ('pandas.core.frame', 'merge'):
            assert isinstance(value_value, MlinspectDataFrame)
            value_value['mlinspect_index_x'] = range(1, len(value_value) + 1)
        self.input_data.append(value_value)

    def before_call_used_args(self, function_info, subscript, call_code, args_code, code_reference, store, args_values):
        """The arguments a function may be called with"""
        # pylint: disable=too-many-arguments
        if function_info == ('pandas.core.frame', 'merge'):
            assert isinstance(args_values[0], MlinspectDataFrame)
            args_values[0]['mlinspect_index_y'] = range(1, len(args_values[0]) + 1)
            self.df_arg = args_values[0]
        elif function_info == ('pandas.core.frame', '__getitem__') and isinstance(args_values, MlinspectSeries):
            self.select = True
        self.before_call_used_args_add_description(args_values, code_reference, function_info, args_code)

    def before_call_used_args_add_description(self, args_values, code_reference, function_info, args_code):
        """Add special descriptions to certain pandas operators"""
        description = None
        if function_info == ('pandas.io.parsers', 'read_csv'):
            filename = args_values[0].split(os.path.sep)[-1]
            description = "{}".format(filename)  # TODO: Add loaded columns as well
        elif function_info == ('pandas.core.frame', 'dropna'):
            description = "dropna"
        elif function_info == ('pandas.core.frame', '__getitem__'):
            # TODO: Can this also be a select?
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
        if function_info == ('pandas.io.parsers', 'read_csv'):
            operator_context = OperatorContext(OperatorType.DATA_SOURCE, function_info)
            return_value = self.execute_inspection_visits_no_parents(operator_context, code_reference,
                                                                     return_value, function_info)
        if function_info == ('pandas.core.groupby.generic', 'agg'):
            operator_context = OperatorContext(OperatorType.GROUP_BY_AGG, function_info)
            return_value = self.execute_inspection_visits_no_parents(operator_context, code_reference,
                                                                     return_value.reset_index(), function_info)
        elif function_info == ('pandas.core.frame', 'dropna'):
            operator_context = OperatorContext(OperatorType.SELECTION, function_info)
            return_value = execute_inspection_visits_unary_operator_df(self, operator_context, code_reference,
                                                                       self.input_data[-1],
                                                                       self.input_data[-1].annotations,
                                                                       return_value)
        elif function_info == ('pandas.core.frame', '__getitem__'):
            # TODO: Can this also be a select
            if self.select:
                self.select = False
                # Gets converted to Selection later?
                operator_context = OperatorContext(OperatorType.SELECTION, function_info)
                return_value = execute_inspection_visits_unary_operator_df(self, operator_context, code_reference,
                                                                           self.input_data[-1],
                                                                           self.input_data[-1].annotations,
                                                                           return_value)
            elif isinstance(return_value, MlinspectDataFrame):
                operator_context = OperatorContext(OperatorType.PROJECTION, function_info)
                return_value['mlinspect_index'] = range(1, len(return_value) + 1)
                return_value = execute_inspection_visits_unary_operator_df(self, operator_context, code_reference,
                                                                           self.input_data[-1],
                                                                           self.input_data[-1].annotations,
                                                                           return_value)
            elif isinstance(return_value, MlinspectSeries):
                operator_context = OperatorContext(OperatorType.PROJECTION, function_info)
                return_value = self.execute_inspection_visits_unary_operator_series(operator_context, code_reference,
                                                                                    return_value,
                                                                                    function_info)
        elif function_info == ('pandas.core.frame', 'groupby'):
            description = self.code_reference_to_description[code_reference]
            return_value.name = description  # TODO: Do not use name here but something else to transport the value
        if function_info == ('pandas.core.frame', 'merge'):
            operator_context = OperatorContext(OperatorType.JOIN, function_info)
            return_value = self.execute_inspection_visits_join_operator_df(operator_context, code_reference,
                                                                           return_value,
                                                                           function_info)

        self.input_data.pop()

        return return_value

    def after_call_used_setkey(self, args_code, value_before, value_after):
        """The value before and after some __setkey__ call"""
        # pylint: disable=unused-argument
        code_reference, function_info, args_code = self.set_key_info
        operator_context = OperatorContext(OperatorType.PROJECTION_MODIFY, function_info)
        value_before['mlinspect_index'] = range(1, len(value_after) + 1)
        execute_inspection_visits_unary_operator_df(self, operator_context, code_reference,
                                                    value_before, value_before.annotations,
                                                    value_after, True)

    def execute_inspection_visits_no_parents(self, operator_context, code_reference, return_value, function_info):
        """Execute inspections when the current operator is a data source and does not have parents in the DAG"""
        # pylint: disable=unused-argument
        annotation_iterators = []
        for inspection in self.inspections:
            iterator_for_inspection = iter_input_data_source(return_value)  # TODO: Create arrays only once
            annotation_iterator = inspection.visit_operator(operator_context, iterator_for_inspection)
            annotation_iterators.append(annotation_iterator)
        return_value = store_inspection_outputs_df(self, annotation_iterators, code_reference, return_value,
                                                   operator_context)
        return return_value

    def store_inspection_outputs_series(self, annotation_iterators, code_reference, return_value, function_info):
        """
        Stores the inspection annotations for the rows in the dataframe and the
        inspection annotations for the DAG operators in a map
        """
        dag_node_identifier = DagNodeIdentifier(self.operator_map[function_info], code_reference,
                                                self.code_reference_to_description.get(code_reference))
        annotations_df = build_annotation_df_from_iters(self.inspections, annotation_iterators)
        annotations_df['mlinspect_index'] = range(1, len(annotations_df) + 1)
        inspection_outputs = {}
        for inspection in self.inspections:
            inspection_outputs[inspection] = inspection.get_operator_annotation_after_visit()
        self.dag_node_identifier_to_inspection_output[dag_node_identifier] = inspection_outputs
        return_value = MlinspectSeries(return_value)
        return_value.annotations = annotations_df
        assert isinstance(return_value, MlinspectSeries)
        return return_value

    def execute_inspection_visits_unary_operator_series(self, operator_context, code_reference, return_value_series,
                                                        function_info):
        """Execute inspections when the current operator has one parent in the DAG"""
        assert isinstance(self.input_data[-1], MlinspectDataFrame)
        assert isinstance(return_value_series, MlinspectSeries)
        annotation_iterators = []
        for inspection in self.inspections:
            inspection_index = self.inspections.index(inspection)
            iterator_for_inspection = iter_input_annotation_output_df_series(inspection_index,
                                                                             self.input_data[-1],
                                                                             self.input_data[-1].annotations,
                                                                             return_value_series)
            annotations_iterator = inspection.visit_operator(operator_context, iterator_for_inspection)
            annotation_iterators.append(annotations_iterator)
        return_value = self.store_inspection_outputs_series(annotation_iterators, code_reference, return_value_series,
                                                            function_info)
        return return_value

    def execute_inspection_visits_join_operator_df(self, operator_context, code_reference, return_value_df,
                                                   function_info):
        """Execute inspections when the current operator has one parent in the DAG"""
        # pylint: disable=unused-argument
        assert "mlinspect_index_x" in return_value_df.columns
        assert "mlinspect_index_y" in return_value_df.columns
        assert isinstance(self.input_data[-1], MlinspectDataFrame)
        assert isinstance(self.df_arg, MlinspectDataFrame)
        annotation_iterators = []
        for inspection in self.inspections:
            inspection_count = len(self.inspections)
            inspection_index = self.inspections.index(inspection)
            iterator_for_inspection = iter_input_annotation_output_df_pair_df(inspection_count,
                                                                              inspection_index,
                                                                              self.input_data[-1],
                                                                              self.input_data[-1].annotations,
                                                                              self.df_arg,
                                                                              self.df_arg.annotations,
                                                                              return_value_df)
            annotations_iterator = inspection.visit_operator(operator_context, iterator_for_inspection)
            annotation_iterators.append(annotations_iterator)
        return_value = store_inspection_outputs_df(self, annotation_iterators, code_reference, return_value_df,
                                                   operator_context)
        return return_value


def execute_inspection_visits_unary_operator_df(backend, operator_context, code_reference, input_data,
                                                input_annotations, return_value_df, appends_col=False):
    """Execute inspections when the current operator has one parent in the DAG"""
    # pylint: disable=too-many-arguments, unused-argument
    assert "mlinspect_index" in return_value_df.columns
    assert isinstance(input_data, MlinspectDataFrame)
    annotation_iterators = []
    for inspection in backend.inspections:
        inspection_count = len(backend.inspections)
        inspection_index = backend.inspections.index(inspection)
        iterator_for_inspection = iter_input_annotation_output_df_df(inspection_count,
                                                                     inspection_index,
                                                                     input_data,
                                                                     input_annotations,
                                                                     return_value_df,
                                                                     appends_col)
        annotations_iterator = inspection.visit_operator(operator_context, iterator_for_inspection)
        annotation_iterators.append(annotations_iterator)
    return_value = store_inspection_outputs_df(backend, annotation_iterators, code_reference, return_value_df,
                                               operator_context)
    return return_value


def iter_input_data_source(output):
    """
    Create an efficient iterator for the inspection input for operators with no parent: Data Source
    """
    output = get_df_row_iterator(output)
    return map(InspectionInputDataSource, output)


def store_inspection_outputs_df(backend, annotation_iterators, code_reference, return_value, operator_context):
    """
    Stores the inspection annotations for the rows in the dataframe and the
    inspection annotations for the DAG operators in a map
    """
    dag_node_identifier = DagNodeIdentifier(operator_context.operator, code_reference,
                                            backend.code_reference_to_description.get(code_reference))
    annotations_df = build_annotation_df_from_iters(backend.inspections, annotation_iterators)
    annotations_df['mlinspect_index'] = range(1, len(annotations_df) + 1)
    inspection_outputs = {}
    for inspection in backend.inspections:
        inspection_outputs[inspection] = inspection.get_operator_annotation_after_visit()
    backend.dag_node_identifier_to_inspection_output[dag_node_identifier] = inspection_outputs
    return_value = MlinspectDataFrame(return_value)
    return_value.annotations = annotations_df
    return_value.backend = backend
    if "mlinspect_index" in return_value.columns:
        return_value = return_value.drop("mlinspect_index", axis=1)
    elif "mlinspect_index_x" in return_value.columns:
        return_value = return_value.drop(["mlinspect_index_x", "mlinspect_index_y"], axis=1)
    assert "mlinspect_index" not in return_value.columns
    assert isinstance(return_value, MlinspectDataFrame)
    return return_value


def iter_input_annotation_output_df_df(inspection_count, inspection_index, input_data, input_annotations, output,
                                       appends_col=False):
    """
    Create an efficient iterator for the inspection input for operators with one parent.
    """
    # pylint: disable=too-many-locals, too-many-arguments
    # Performance tips:
    # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
    data_before_with_annotations = pandas.merge(input_data, input_annotations, left_on="mlinspect_index",
                                                right_on="mlinspect_index")
    joined_df = pandas.merge(data_before_with_annotations, output, left_on="mlinspect_index",
                             right_on="mlinspect_index")

    column_index_input_end = len(input_data.columns)
    column_annotation_current_inspection = column_index_input_end + inspection_index
    column_index_annotation_end = column_index_input_end + inspection_count

    input_df_view = joined_df.iloc[:, 0:column_index_input_end - 1]
    input_df_view.columns = input_data.columns[0:-1]

    annotation_df_view = joined_df.iloc[:,
                                        column_annotation_current_inspection:column_annotation_current_inspection + 1]

    output_df_view = joined_df.iloc[:, column_index_annotation_end:]
    if not appends_col:
        output_df_view.columns = output.columns[0:-1]
    else:  # e.g., __setkey__  can add columns, but they are then behind mlinspect_index
        output_df_view.columns = list(output.columns[0:-2]) + list(output.columns[-1:])

    input_rows = get_df_row_iterator(input_df_view)
    annotation_rows = get_df_row_iterator(annotation_df_view)
    output_rows = get_df_row_iterator(output_df_view)

    return map(lambda input_tuple: InspectionInputUnaryOperator(*input_tuple),
               zip(input_rows, annotation_rows, output_rows))


def iter_input_annotation_output_df_pair_df(inspection_count, inspection_index, x_data, x_annotations, y_data,
                                            y_annotations, output):
    """
    Create an efficient iterator for the inspection input for operators with one parent.
    """
    # pylint: disable=too-many-locals, too-many-arguments
    # Performance tips:
    # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
    x_before_with_annotations = pandas.merge(x_data, x_annotations, left_on="mlinspect_index_x",
                                             right_on="mlinspect_index", suffixes=["_x_data", "_x_annot"])
    y_before_with_annotations = pandas.merge(y_data, y_annotations, left_on="mlinspect_index_y",
                                             right_on="mlinspect_index", suffixes=["_y_data", "_y_annot"])
    df_x_output = pandas.merge(x_before_with_annotations, output, left_on="mlinspect_index_x",
                               right_on="mlinspect_index_x", suffixes=["_x", "_output"])
    df_x_output_y = pandas.merge(df_x_output, y_before_with_annotations, left_on="mlinspect_index_y",
                                 right_on="mlinspect_index_y", suffixes=["_x_output", "_y_output"])

    column_index_x_end = len(x_data.columns)
    column_annotation_x_current_inspection = column_index_x_end + inspection_index
    column_index_output_start = column_index_x_end + inspection_count
    column_index_y_start = column_index_output_start + len(output.columns) - 2
    column_index_y_end = column_index_y_start + len(y_data.columns) - 1
    column_annotation_y_current_inspection = column_index_y_end + inspection_index

    df_x_output_y = df_x_output_y.drop(['mlinspect_index_x_output', 'mlinspect_index_y'], axis=1)

    input_x_view = df_x_output_y.iloc[:, 0:column_index_x_end - 1]
    input_x_view.columns = x_data.columns[0:-1]
    annotation_x_view = df_x_output_y.iloc[:, column_annotation_x_current_inspection:
                                           column_annotation_x_current_inspection + 1]
    annotation_x_view.columns = [annotation_x_view.columns[0].replace("_x_output", "")]

    output_df_view = df_x_output_y.iloc[:, column_index_output_start:column_index_y_start]
    output_df_view.columns = [column for column in output.columns if
                              (column not in ("mlinspect_index_x", "mlinspect_index_y"))]

    input_y_view = df_x_output_y.iloc[:, column_index_y_start:column_index_y_end]
    input_y_view.columns = y_data.columns[0:-1]
    annotation_y_view = df_x_output_y.iloc[:, column_annotation_y_current_inspection:
                                           column_annotation_y_current_inspection + 1]
    annotation_y_view.columns = [annotation_y_view.columns[0].replace("_y_output", "")]

    input_iterators = []
    annotation_iterators = []

    input_iterators.append(get_df_row_iterator(input_x_view))
    annotation_iterators.append(get_df_row_iterator(annotation_x_view))

    input_iterators.append(get_df_row_iterator(input_y_view))
    annotation_iterators.append(get_df_row_iterator(annotation_y_view))

    input_rows = map(list, zip(*input_iterators))
    annotation_rows = map(list, zip(*annotation_iterators))

    output_rows = get_df_row_iterator(output_df_view)

    return map(lambda input_tuple: InspectionInputNAryOperator(*input_tuple),
               zip(input_rows, annotation_rows, output_rows))


def iter_input_annotation_output_df_series(inspection_index, input_data, input_annotations, output):
    """
    Create an efficient iterator for the inspection input for operators with one parent.
    """
    # pylint: disable=too-many-locals
    # Performance tips:
    # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
    data_before_with_annotations = pandas.merge(input_data, input_annotations, left_on="mlinspect_index",
                                                right_on="mlinspect_index")

    column_index_input_end = len(input_data.columns)
    column_annotation_current_inspection = column_index_input_end + inspection_index

    input_df_view = data_before_with_annotations.iloc[:, 0:column_index_input_end - 1]
    input_df_view.columns = input_data.columns[0:-1]

    annotation_df_view = data_before_with_annotations.iloc[:, column_annotation_current_inspection:
                                                           column_annotation_current_inspection + 1]

    input_rows = get_df_row_iterator(input_df_view)
    annotation_rows = get_df_row_iterator(annotation_df_view)
    output_rows = get_series_row_iterator(output)

    return map(lambda input_tuple: InspectionInputUnaryOperator(*input_tuple),
               zip(input_rows, annotation_rows, output_rows))
