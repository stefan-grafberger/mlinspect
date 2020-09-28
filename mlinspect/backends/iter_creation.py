"""
Functions to create the iterators for the inspections
"""
import itertools

import pandas

from mlinspect.backends.backend_utils import get_df_row_iterator, get_iterator_for_type, get_annotation_rows
from mlinspect.inspections.inspection_input import InspectionInputDataSource, InspectionInputUnaryOperator, \
    InspectionInputNAryOperator, InspectionInputSinkOperator


def iter_input_data_source(inspection_count, output, operator_context):
    """
    Create an efficient iterator for the inspection input for operators with no parent: Data Source
    """
    output_column_info, output_rows = get_df_row_iterator(output)
    duplicated_output_iterators = itertools.tee(output_rows, inspection_count)
    inspection_iterators = []
    for inspection_index in range(inspection_count):
        output_iterator = duplicated_output_iterators[inspection_index]
        inspection_iterator = InspectionInputDataSource(operator_context, output_column_info, output_iterator)
        inspection_iterators.append(inspection_iterator)

    return inspection_iterators


def iter_input_annotation_output_map(inspection_count, input_data, input_annotations, output):
    """
    Create an efficient iterator for the inspection input for operators with one parent that do not
    change the row order.
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
    Create an efficient iterator for the inspection input for operators with one parent that do change the
    row order or drop some rows, like selections.
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


def iter_input_annotation_output_join(inspection_count, x_data, x_annotations, y_data,
                                      y_annotations, output):
    """
    Create an efficient iterator for the inspection input for join operators.
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


def iter_input_annotation_output_nary_op(inspection_count, transformer_data_with_annotations, output_data):
    """
    Create an efficient iterator for the inspection input for operators with multiple parents that do
    not change the order of rows or remove rows: concatenations.
    """
    # pylint: disable=too-many-locals
    input_iterators = []
    for input_data, _ in transformer_data_with_annotations:
        input_iterators.append(get_iterator_for_type(input_data, True))
    input_rows = map(list, zip(*input_iterators))
    duplicated_input_iterators = itertools.tee(input_rows, inspection_count)

    output_rows = get_iterator_for_type(output_data, False)
    duplicated_output_iterators = itertools.tee(output_rows, inspection_count)

    inspection_iterators = []
    for inspection_index in range(inspection_count):
        annotation_iterators = []
        for _, annotations in transformer_data_with_annotations:
            annotation_iterators.append(get_annotation_rows(annotations, inspection_index))
        annotation_rows = map(list, zip(*annotation_iterators))
        input_iterator = duplicated_input_iterators[inspection_index]
        output_iterator = duplicated_output_iterators[inspection_index]
        inspection_iterator = map(lambda input_tuple: InspectionInputNAryOperator(*input_tuple),
                                  zip(input_iterator, annotation_rows, output_iterator))
        inspection_iterators.append(inspection_iterator)

    return inspection_iterators


def iter_input_annotation_output_sink_op(inspection_count, data, target):
    """
    Create an efficient iterator for the inspection input when there is no output, e.g., estimators.
    """
    # pylint: disable=too-many-locals
    input_iterators = [get_iterator_for_type(data, False), get_iterator_for_type(target, True)]
    input_rows = map(list, zip(*input_iterators))
    duplicated_input_iterators = itertools.tee(input_rows, inspection_count)

    inspection_iterators = []
    for inspection_index in range(inspection_count):
        input_iterator = duplicated_input_iterators[inspection_index]
        annotation_iterators = [get_annotation_rows(data.annotations, inspection_index),
                                get_annotation_rows(target.annotations, inspection_index)]
        annotation_rows = map(list, zip(*annotation_iterators))
        inspection_iterator = map(lambda input_tuple: InspectionInputSinkOperator(*input_tuple),
                                  zip(input_iterator, annotation_rows))
        inspection_iterators.append(inspection_iterator)

    return inspection_iterators
