"""
Functions to create the iterators for the inspections
"""
import itertools
from typing import List

import pandas

from mlinspect.backends._backend import AnnotatedDfObject
from mlinspect.backends._backend_utils import get_df_row_iterator, get_iterator_for_type, get_annotation_rows
from mlinspect.inspections._inspection_input import InspectionInputDataSource, InspectionInputUnaryOperator, \
    InspectionInputNAryOperator, InspectionInputSinkOperator, InspectionRowDataSource, InspectionRowUnaryOperator, \
    InspectionRowNAryOperator, ColumnInfo, InspectionRowSinkOperator


def iter_input_data_source(inspection_count, output, operator_context, non_data_function_args):
    """
    Create an efficient iterator for the inspection input for operators with no parent: Data Source
    """
    if inspection_count == 0:
        return []
    output_columns, output_rows = get_iterator_for_type(output)
    duplicated_output_iterators = itertools.tee(output_rows, inspection_count)
    inspection_iterators = []
    for inspection_index in range(inspection_count):
        output_iterator = duplicated_output_iterators[inspection_index]
        row_iterator = map(InspectionRowDataSource, output_iterator)
        inspection_iterator = InspectionInputDataSource(operator_context, output_columns, row_iterator, non_data_function_args)
        inspection_iterators.append(inspection_iterator)

    return inspection_iterators


def iter_input_annotation_output_map(inspection_count, input_data, input_annotations, output, operator_context,
                                     non_data_function_args, columns=None):
    """
    Create an efficient iterator for the inspection input for operators with one parent that do not
    change the row order.
    """
    # pylint: disable=too-many-locals, too-many-arguments
    if inspection_count == 0:
        return []

    input_columns, input_rows = get_iterator_for_type(input_data, True)
    output_columns, output_rows = get_iterator_for_type(output, False, columns)
    duplicated_input_iterators = itertools.tee(input_rows, inspection_count)
    duplicated_output_iterators = itertools.tee(output_rows, inspection_count)

    inspection_iterators = []
    for inspection_index in range(inspection_count):
        input_iterator = duplicated_input_iterators[inspection_index]
        output_iterator = duplicated_output_iterators[inspection_index]
        annotation_rows = get_annotation_rows(input_annotations, inspection_index)
        row_iterator = map(lambda input_tuple: InspectionRowUnaryOperator(*input_tuple),
                           zip(input_iterator, annotation_rows, output_iterator))
        inspection_iterator = InspectionInputUnaryOperator(operator_context, input_columns, output_columns,
                                                           row_iterator, non_data_function_args)
        inspection_iterators.append(inspection_iterator)

    return inspection_iterators


def iter_input_annotation_output_resampled(inspection_count, input_data, input_annotations, output, operator_context,
                                           non_data_function_args):
    """
    Create an efficient iterator for the inspection input for operators with one parent that do change the
    row order or drop some rows, like selections.
    """
    # pylint: disable=too-many-locals, too-many-arguments
    if inspection_count == 0:
        return []

    data_before_with_annotations = pandas.concat([input_data.reset_index(drop=True), input_annotations], axis=1)
    joined_df = output.merge(data_before_with_annotations, left_on="mlinspect_index",
                             right_on="mlinspect_index")

    # After these operations, joined_df contains the following columns from left to right:
    # output columns
    # mlinspect_index
    # input_data columns
    # input_annotations columns

    column_index_output_end = len(output.columns)
    output_df_view = joined_df.iloc[:, 0:column_index_output_end - 1]  # -1 excludes the mlinspect_index
    output_df_view.columns = output.columns[0:-1] # -1 excludes the mlinspect_index
    output_columns, output_rows = get_df_row_iterator(output_df_view)
    duplicated_output_iterators = itertools.tee(output_rows, inspection_count)

    column_index_input_end = column_index_output_end + len(input_data.columns) - 1  # -1 excludes the mlinspect_index

    input_df_view = joined_df.iloc[:, column_index_output_end:column_index_input_end]
    input_df_view.columns = input_data.columns[0:-1]  # -1 excludes the mlinspect_index
    input_columns, input_rows = get_df_row_iterator(input_df_view)
    duplicated_input_iterators = itertools.tee(input_rows, inspection_count)

    inspection_iterators = []
    for inspection_index in range(inspection_count):
        input_iterator = duplicated_input_iterators[inspection_index]
        output_iterator = duplicated_output_iterators[inspection_index]
        column_annotation_current_inspection = column_index_input_end + inspection_index
        annotation_rows = get_annotation_rows(joined_df, column_annotation_current_inspection)
        row_iterator = map(lambda input_tuple: InspectionRowUnaryOperator(*input_tuple),
                           zip(input_iterator, annotation_rows, output_iterator))
        inspection_iterator = InspectionInputUnaryOperator(operator_context, input_columns, output_columns,
                                                           row_iterator, non_data_function_args)
        inspection_iterators.append(inspection_iterator)

    return inspection_iterators


def iter_input_annotation_output_join(inspection_count, x_data, x_annotations, y_data,
                                      y_annotations, output, operator_context, non_data_function_args):
    """
    Create an efficient iterator for the inspection input for join operators.
    """
    # pylint: disable=too-many-locals, too-many-arguments
    if inspection_count == 0:
        return []

    x_before_with_annotations = pandas.concat([x_data.reset_index(drop=True), x_annotations], axis=1)
    y_before_with_annotations = pandas.concat([y_data.reset_index(drop=True), y_annotations], axis=1)
    df_x_output = output.merge(x_before_with_annotations, on="mlinspect_index_x", suffixes=["_output", "_x"])
    df_x_output_y = df_x_output.merge(y_before_with_annotations, on="mlinspect_index_y",
                                      suffixes=["_x_output", "_y_output"])

    df_x_output_y = df_x_output_y.drop(['mlinspect_index_y', 'mlinspect_index_x'], axis=1)

    # After these operations, df_x_output_y contains the following columns from left to right:
    # output columns
    # x_data columns
    # x_annotations columns
    # y_data columns
    # y_annotations columns

    column_index_output_start = 0
    column_index_output_end = len(output.columns) - 2  # -2 accounts for the index columns

    column_index_x_start = column_index_output_end
    column_index_x_end = column_index_x_start + len(x_data.columns) - 1  # -1 accounts for mlinspect_index_x
    column_index_y_start = column_index_x_end + inspection_count
    column_index_y_end = column_index_y_start + len(y_data.columns) - 1  # -1 accounts for mlinspect_index_y

    input_x_view = df_x_output_y.iloc[:, column_index_x_start:column_index_x_end]
    input_x_view.columns = x_data.columns[0:-1]  # -1 accounts for mlinspect_index_x
    input_y_view = df_x_output_y.iloc[:, column_index_y_start:column_index_y_end]
    input_y_view.columns = y_data.columns[0:-1]  # -1 accounts for mlinspect_index_y
    input_x_columns, input_x_iterator = get_df_row_iterator(input_x_view)
    assert isinstance(input_x_columns, ColumnInfo)
    input_y_columns, input_y_iterator = get_df_row_iterator(input_y_view)
    assert isinstance(input_y_columns, ColumnInfo)
    input_rows = map(tuple, zip(input_x_iterator, input_y_iterator))
    inputs_columns = [input_x_columns, input_y_columns]
    duplicated_input_iterators = itertools.tee(input_rows, inspection_count)

    output_df_view = df_x_output_y.iloc[:, column_index_output_start:column_index_output_end]
    output_df_view.columns = [column for column in output.columns if
                              (column not in ("mlinspect_index_x", "mlinspect_index_y"))]
    output_columns, output_rows = get_df_row_iterator(output_df_view)
    duplicated_output_iterators = itertools.tee(output_rows, inspection_count)

    inspection_iterators = []
    for inspection_index in range(inspection_count):
        input_iterator = duplicated_input_iterators[inspection_index]
        output_iterator = duplicated_output_iterators[inspection_index]

        column_annotation_y_current_inspection = column_index_y_end + inspection_index
        column_annotation_x_current_inspection = column_index_x_end + inspection_index
        annotation_iterators = [get_annotation_rows(df_x_output_y, column_annotation_x_current_inspection),
                                get_annotation_rows(df_x_output_y, column_annotation_y_current_inspection)]

        annotation_rows = map(tuple, zip(*annotation_iterators))

        row_iterator = map(lambda input_tuple: InspectionRowNAryOperator(*input_tuple),
                           zip(input_iterator, annotation_rows, output_iterator))
        inspection_iterator = InspectionInputNAryOperator(operator_context, inputs_columns,
                                                          output_columns, row_iterator, non_data_function_args)
        inspection_iterators.append(inspection_iterator)

    return inspection_iterators


def iter_input_annotation_output_nary_op(inspection_count, annotated_inputs: List[AnnotatedDfObject], output_data,
                                         operator_context, non_data_function_args):
    """
    Create an efficient iterator for the inspection input for operators with multiple parents that do
    not change the order of rows or remove rows: concatenations.
    """
    # pylint: disable=too-many-locals
    if inspection_count == 0:
        return []

    input_iterators = []
    inputs_columns = []
    for annotated_input in annotated_inputs:
        column_info, row_iterator = get_iterator_for_type(annotated_input.result_data, True)
        inputs_columns.append(column_info)
        input_iterators.append(row_iterator)
    input_rows = map(list, zip(*input_iterators))
    duplicated_input_iterators = itertools.tee(input_rows, inspection_count)

    output_columns, output_rows = get_iterator_for_type(output_data, False)
    duplicated_output_iterators = itertools.tee(output_rows, inspection_count)

    inspection_iterators = []
    for inspection_index in range(inspection_count):
        annotation_iterators = []
        for annotated_input in annotated_inputs:
            annotation_iterators.append(get_annotation_rows(annotated_input.result_annotation, inspection_index))
        annotation_rows = map(list, zip(*annotation_iterators))
        input_iterator = duplicated_input_iterators[inspection_index]
        output_iterator = duplicated_output_iterators[inspection_index]
        row_iterator = map(lambda input_tuple: InspectionRowNAryOperator(*input_tuple),
                           zip(input_iterator, annotation_rows, output_iterator))
        inspection_iterator = InspectionInputNAryOperator(operator_context, inputs_columns,
                                                          output_columns, row_iterator, non_data_function_args)
        inspection_iterators.append(inspection_iterator)

    return inspection_iterators


def iter_input_annotation_output_sink_op(inspection_count, data, data_annotation, target, target_annotation,
                                         operator_context, non_data_function_args):
    """
    Create an efficient iterator for the inspection input when there is no output, e.g., estimators.
    """
    # pylint: disable=too-many-locals, too-many-arguments
    if inspection_count == 0:
        return []

    input_data_columns, input_data_iterators = get_iterator_for_type(data, False)
    input_target_columns, input_target_iterators = get_iterator_for_type(target, True)
    inputs_columns = [input_data_columns, input_target_columns]
    input_rows = map(tuple, zip(input_data_iterators, input_target_iterators))
    duplicated_input_iterators = itertools.tee(input_rows, inspection_count)

    inspection_iterators = []
    for inspection_index in range(inspection_count):
        input_iterator = duplicated_input_iterators[inspection_index]
        annotation_iterators = [get_annotation_rows(data_annotation, inspection_index),
                                get_annotation_rows(target_annotation, inspection_index)]
        annotation_rows = map(tuple, zip(*annotation_iterators))
        row_iterator = map(lambda input_tuple: InspectionRowSinkOperator(*input_tuple),
                           zip(input_iterator, annotation_rows))
        inspection_iterator = InspectionInputSinkOperator(operator_context, inputs_columns, row_iterator,
                                                          non_data_function_args)
        inspection_iterators.append(inspection_iterator)

    return inspection_iterators
