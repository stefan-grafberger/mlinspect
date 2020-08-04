"""
The pandas backend
"""
import os
from functools import partial

import pandas
from pandas import DataFrame

from mlinspect.instrumentation.analyzer_input import AnalyzerInputUnaryOperator, AnalyzerInputRow, \
    AnalyzerInputDataSource
from mlinspect.instrumentation.analyzers.print_first_rows_analyzer import PrintFirstRowsAnalyzer
from mlinspect.instrumentation.backends.backend import Backend


class PandasBackend(Backend):
    """
    The pandas backend
    """

    prefix = "pandas"

    def __init__(self):
        super().__init__()
        self.input_data = None
        self.input_annotations = None

    def before_call_used_value(self, function_info, subscript, call_code, value_code, value_value, ast_lineno,
                               ast_col_offset):
        """The value or module a function may be called on"""
        # pylint: disable=too-many-arguments, unused-argument, no-self-use
        print("pandas_before_call_used_value")

        if function_info == ('pandas.core.frame', 'dropna'):
            print("dropna")
            value_value['mlinspect_index'] = range(1, len(value_value) + 1)
            self.input_data = value_value


    def before_call_used_args(self, function_info, subscript, call_code, args_code, ast_lineno, ast_col_offset,
                              args_values):
        """The arguments a function may be called with"""
        # pylint: disable=too-many-arguments, unused-argument
        self.before_call_used_args_add_description(args_values, ast_col_offset, ast_lineno, function_info)

    def before_call_used_args_add_description(self, args_values, ast_col_offset, ast_lineno, function_info):
        """Add special descriptions to certain pandas operators"""
        description = None
        if function_info == ('pandas.io.parsers', 'read_csv'):
            filename = args_values[0].split(os.path.sep)[-1]
            description = "{}".format(filename)
        elif function_info == ('pandas.core.frame', 'dropna'):
            description = "dropna"
        elif function_info == ('pandas.core.frame', '__getitem__'):
            # TODO: Can this also be a select?
            key_arg = args_values[0].split(os.path.sep)[-1]
            description = "to {}".format([key_arg])
        if description:
            self.call_description_map[(ast_lineno, ast_col_offset)] = description

    def before_call_used_kwargs(self, function_info, subscript, call_code, kwargs_code, ast_lineno, ast_col_offset,
                                kwargs_values):
        """The keyword arguments a function may be called with"""
        # pylint: disable=too-many-arguments, unused-argument, no-self-use
        print("pandas_before_call_used_kwargs")

    def after_call_used(self, function_info, subscript, call_code, return_value, ast_lineno, ast_col_offset):
        """The return value of some function"""
        # pylint: disable=too-many-arguments, unused-argument, no-self-use
        print("pandas_after_call_used")
        if function_info == ('pandas.io.parsers', 'read_csv'):
            analyzer = PrintFirstRowsAnalyzer(5)
            print("read.csv:")
            # Performance tips:
            # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
            # We need our own iterator type:
            # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas/41022840#41022840
            annotations_iterator = analyzer.visit_operator("Data Source", iter_input_data_source(return_value))
            annotations_df = DataFrame(annotations_iterator, columns=["TestAnalyzer"])
            annotations_df['mlinspect_index'] = range(1, len(annotations_df) + 1)
            self.input_annotations = annotations_df

        elif function_info == ('pandas.core.frame', 'dropna'):
            assert "mlinspect_index" in return_value.columns
            analyzer = PrintFirstRowsAnalyzer(5)
            print("dropna")
            annotations_iterator = analyzer.visit_operator("Selection",
                                                           iter_input_annotation_output(self.input_data,
                                                                                        self.input_annotations,
                                                                                        return_value))
            annotations_df = DataFrame(annotations_iterator, columns=["TestAnalyzer"])
            self.input_data = None
            self.input_annotations = annotations_df
            return_value = return_value.drop("mlinspect_index", axis=1)
            assert "mlinspect_index" not in return_value.columns


def iter_input_data_source(output):
    """
    Create an efficient iterator for the analyzer input for operators with no parent: Data Source
    """
    output = get_row_iterator(output, 0, len(output.columns))
    return map(AnalyzerInputDataSource, output)


def iter_input_annotation_output(input_data, input_annotations, output):
    """
    Create an efficient iterator for the analyzer input for operators with one parent.
    """
    # Performance tips:
    # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
    # We need our own iterator type:
    # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas/41022840#41022840
    column_index_input_end = len(input_data.columns)
    column_index_annotation_end = column_index_input_end + 1
    data_before_with_annotations = pandas.merge(input_data, input_annotations, left_on="mlinspect_index",
                                                right_on="mlinspect_index")
    joined_df = pandas.merge(data_before_with_annotations, output, left_on="mlinspect_index",
                             right_on="mlinspect_index")

    input_rows = get_row_iterator(joined_df, 0, column_index_input_end)
    annotation_rows = get_row_iterator(joined_df, column_index_input_end,
                                       column_index_annotation_end)
    output_rows = get_row_iterator(joined_df, column_index_annotation_end,
                                   len(joined_df.columns))

    return map(lambda input_tuple: AnalyzerInputUnaryOperator(*input_tuple),
               zip(input_rows, annotation_rows, output_rows))


def get_row_iterator(joined_df, start_col, end_col):
    """
    Create an efficient iterator for the data frame rows.
    The implementation is inspired by the implementation of the pandas DataFrame.itertuple method
    """
    arrays = []
    fields = list(joined_df.columns[start_col:end_col])
    # use integer indexing because of possible duplicate column names
    arrays.extend(joined_df.iloc[:, k] for k in range(start_col, end_col))  # sort_index fixes things

    partial_func_create_row = partial(AnalyzerInputRow, fields=fields)
    test = map(partial_func_create_row, zip(*arrays))
    return test
