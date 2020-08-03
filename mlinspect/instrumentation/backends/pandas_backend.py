"""
The pandas backend
"""
import collections
import os
from collections import namedtuple

import pandas
from pandas.compat import PY37
from pandas import DataFrame

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
            analyzer = PrintFirstRowsAnalyzer(5)
            print("dropna")
            # .copy() necessary if inplace. However, inplace should not be used and
            # we ignore this edge case for now
            self.input_data = value_value

    def before_call_used_args(self, function_info, subscript, call_code, args_code, ast_lineno, ast_col_offset,
                              args_values):
        """The arguments a function may be called with"""
        # pylint: disable=too-many-arguments, unused-argument
        self.before_call_used_args_add_description(args_values, ast_col_offset, ast_lineno, function_info)

    def before_call_used_args_add_description(self, args_values, ast_col_offset, ast_lineno, function_info):
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
            annotations_iterator = analyzer.visit_operator(return_value.sort_index().itertuples())
            annotations_df = DataFrame(annotations_iterator, columns=[str(analyzer)])
            print(return_value.sort_index())
            self.input_annotations = annotations_df
        elif function_info == ('pandas.core.frame', 'dropna'):
            analyzer = PrintFirstRowsAnalyzer(5)
            print("dropna")
            # Performance tips:
            # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
            # We need our own iterator type:
            # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas/41022840#41022840
            annotations_iterator = analyzer.visit_operator(iter_input_annotation_output(self.input_data,
                                                                                        self.input_annotations,
                                                                                        return_value))
            annotations_df = DataFrame(annotations_iterator, columns=["TestAnalyzer"])
            # print(annotations_df)
            self.input_data = None
            self.input_annotations = None


def iter_input_annotation_output(input_data, input_annotations, output):
    column_index_input_end = len(input_data.columns)
    column_index_annotation_end = column_index_input_end + len(input_annotations.columns)
    data_before_with_annotations = pandas.concat([input_data, input_annotations], axis=1)
    joined_df = pandas.merge(data_before_with_annotations, output, left_index=True, right_index=True)

    input = get_named_tuple_for_tuple_part(joined_df, "input", 0, column_index_input_end)
    annotations = get_named_tuple_for_tuple_part(joined_df, "annotations", column_index_input_end,
                                                 column_index_annotation_end)
    output = get_named_tuple_for_tuple_part(joined_df, "output", column_index_annotation_end,
                                            len(joined_df.columns))
    # this is an adjusted version of the pandas DataFrame itertuples method
    # split into multiple parts
    itertuple = collections.namedtuple("AnalyzerInput", ["input", "annotations", "output"], rename=True)
    return map(itertuple._make, zip(input, annotations, output))


def get_named_tuple_for_tuple_part(joined_df, name, start_col, end_col):
    arrays = []
    fields = list(joined_df.columns[start_col:end_col])
    # use integer indexing because of possible duplicate column names
    arrays.extend(joined_df.iloc[:, k] for k in range(start_col, end_col))
    # Python versions before 3.7 support at most 255 arguments to constructors
    can_return_named_tuples = PY37 or len(joined_df.columns) + joined_df.index < 255
    if can_return_named_tuples:
        itertuple = collections.namedtuple(name, fields, rename=True)
        return map(itertuple._make, zip(*arrays))
    # fallback to regular tuples
    return zip(*arrays)
