"""
The pandas backend
"""
import os
from functools import partial

import pandas
from pandas import DataFrame

from mlinspect.instrumentation.analyzer_input import AnalyzerInputUnaryOperator, AnalyzerInputRow, \
    AnalyzerInputDataSource
from mlinspect.instrumentation.backends.backend import Backend
from mlinspect.instrumentation.backends.pandas_backend_frame_wrapper import MlinspectDataFrame


class PandasBackend(Backend):
    """
    The pandas backend
    """

    prefix = "pandas"

    def __init__(self):
        super().__init__()
        self.input_data = None

    def before_call_used_value(self, analyzers, function_info, subscript, call_code, value_code, value_value,
                               ast_lineno, ast_col_offset):
        """The value or module a function may be called on"""
        # pylint: disable=too-many-arguments, unused-argument, no-self-use
        if function_info == ('pandas.core.frame', 'dropna'):
            assert isinstance(value_value, MlinspectDataFrame)
            value_value['mlinspect_index'] = range(1, len(value_value) + 1)
            self.input_data = value_value

    def before_call_used_args(self, analyzers, function_info, subscript, call_code, args_code, ast_lineno,
                              ast_col_offset, args_values):
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

    def before_call_used_kwargs(self, analyzers, function_info, subscript, call_code, kwargs_code, ast_lineno,
                                ast_col_offset, kwargs_values):
        """The keyword arguments a function may be called with"""
        # pylint: disable=too-many-arguments, unused-argument, no-self-use, unnecessary-pass
        pass

    def after_call_used(self, analyzers, function_info, subscript, call_code, return_value, ast_lineno,
                        ast_col_offset):
        """The return value of some function"""
        # pylint: disable=too-many-arguments, unused-argument, no-self-use, too-many-locals
        if function_info == ('pandas.io.parsers', 'read_csv'):
            # Performance tips:
            # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
            # We need our own iterator type:
            # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas/41022840#41022840
            return_value = self.execute_analyzer_visits_data_source(analyzers, ast_col_offset, ast_lineno, return_value)
        elif function_info == ('pandas.core.frame', 'dropna'):
            operator_name = "Selection"
            return_value = self.execute_analyzer_visits_unary_operator(analyzers, operator_name, ast_col_offset,
                                                                       ast_lineno, return_value)

        return return_value

    def execute_analyzer_visits_data_source(self, analyzers, ast_col_offset, ast_lineno, return_value):
        """Execute analyzers when the current operator is a data source and does not have parents in the DAG"""
        annotation_iterators = []
        for analyzer in analyzers:
            annotation_iterator = analyzer.visit_operator("Data Source", iter_input_data_source(return_value))
            annotation_iterators.append(annotation_iterator)
        return_value = self.store_analyzer_outputs(analyzers, annotation_iterators, ast_col_offset, ast_lineno,
                                                   return_value)
        return return_value

    def store_analyzer_outputs(self, analyzers, annotation_iterators, ast_col_offset, ast_lineno, return_value):
        """
        Stores the analyzer annotations for the rows in the dataframe and the
        analyzer annotations for the DAG operators in a map
        """
        # pylint: disable=too-many-arguments
        annotation_iterators = zip(*annotation_iterators)
        analyzer_names = [str(analyzer) for analyzer in analyzers]
        annotations_df = DataFrame(annotation_iterators, columns=analyzer_names)
        annotations_df['mlinspect_index'] = range(1, len(annotations_df) + 1)
        analyzer_outputs = {}
        for analyzer in analyzers:
            analyzer_output = analyzer.get_operator_annotation_after_visit()
            print(analyzer_output)
            analyzer_outputs[analyzer] = analyzer_output
        self.call_analyzer_output_map[(ast_lineno, ast_col_offset)] = analyzer_outputs
        return_value = MlinspectDataFrame(return_value)
        return_value.annotations = annotations_df
        self.input_data = None
        if "mlinspect_index" in return_value.columns:
            return_value = return_value.drop("mlinspect_index", axis=1)
        assert "mlinspect_index" not in return_value.columns
        assert isinstance(return_value, MlinspectDataFrame)
        return return_value

    def execute_analyzer_visits_unary_operator(self, analyzers, operator_name, ast_col_offset, ast_lineno,
                                               return_value):
        """Execute analyzers when the current operator has one parent in the DAG"""
        # pylint: disable=too-many-arguments
        assert "mlinspect_index" in return_value.columns
        assert isinstance(self.input_data, MlinspectDataFrame)
        annotation_iterators = []
        for analyzer in analyzers:
            annotations_iterator = analyzer.visit_operator(operator_name,
                                                           iter_input_annotation_output(self.input_data,
                                                                                        self.input_data.annotations,
                                                                                        return_value))
            annotation_iterators.append(annotations_iterator)
        return_value = self.store_analyzer_outputs(analyzers, annotation_iterators, ast_col_offset, ast_lineno,
                                                   return_value)
        return return_value


def iter_input_data_source(output):
    """
    Create an efficient iterator for the analyzer input for operators with no parent: Data Source
    """
    output = get_row_iterator(output)
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

    # TODO: Add analyzer output to the extracted DAG instead of printing it
    # TODO: Do not always use the print test analazyer, but build a test case
    # TODO: and add functions/arguments to inspector and executor.
    # TODO: Then support the rest of the pandas functions for this example.
    # TODO: Sklearn backend as part of next PR.
    # TODO: Move SklearnWirPreprocessor functionality to backend interface
    # TODO: Vertex classes as data classes. also: maybe rename to node
    # TODO: In WirToDagTransformer the map to operators should also be moved into backend.
    # TODO: Then we can also introduce warnings whenever there is e.g., a pandas function
    # TODO: that the pandas backend can not deal with (has no operator mapping for)
    # TODO: Add utility function to extract the library name, pandas and sklearn etc.
    # TODO: extract the function info adjustments for overwritten classes into backend in some way

    # FIXME: When there are multiple analyzers, each one should see its own annotations only

    input_df_view = joined_df.iloc[:, 0:column_index_input_end-1]
    input_df_view.columns = input_data.columns[0:-1]

    annotation_df_view = joined_df.iloc[:, column_index_input_end:column_index_annotation_end+1]

    output_df_view = joined_df.iloc[:, column_index_annotation_end:]
    output_df_view.columns = output.columns[0:-1]

    input_rows = get_row_iterator(input_df_view)
    annotation_rows = get_row_iterator(annotation_df_view)
    output_rows = get_row_iterator(output_df_view)

    return map(lambda input_tuple: AnalyzerInputUnaryOperator(*input_tuple),
               zip(input_rows, annotation_rows, output_rows))


def get_row_iterator(dataframe):
    """
    Create an efficient iterator for the data frame rows.
    The implementation is inspired by the implementation of the pandas DataFrame.itertuple method
    """
    arrays = []
    fields = list(dataframe.columns)
    # use integer indexing because of possible duplicate column names
    arrays.extend(dataframe.iloc[:, k] for k in range(0, len(dataframe.columns)))

    partial_func_create_row = partial(AnalyzerInputRow, fields=fields)
    test = map(partial_func_create_row, zip(*arrays))
    return test
