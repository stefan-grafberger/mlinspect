"""
The pandas backend
"""
import os
from collections import namedtuple

import pandas
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

            def iter_input_annotation_output():
                column_index_input_end = len(self.input_data.columns) + 1
                column_index_annotation_end = column_index_input_end + len(self.input_annotations.columns)
                data_before_with_annotations = pandas.concat([self.input_data, self.input_annotations], axis=1)
                joined_df = pandas.merge(data_before_with_annotations, return_value, left_index=True, right_index=True)
                for row in joined_df.itertuples():
                    AnalyzerInputRow = namedtuple('Row', ['input', 'annotation', 'output'])
                    next_row = AnalyzerInputRow(input=row[0:column_index_input_end],
                                                annotation=row[column_index_input_end:column_index_annotation_end],
                                                output=row[column_index_annotation_end:])
                    yield next_row

            annotations_iterator = analyzer.visit_operator(iter_input_annotation_output())
            annotations_df = DataFrame(annotations_iterator, columns=[str(analyzer)])
            # print(annotations_df)
            self.input_data = None
            self.input_annotations = None
