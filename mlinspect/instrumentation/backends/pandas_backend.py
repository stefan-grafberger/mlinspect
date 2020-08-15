"""
The pandas backend
"""
import os
from functools import partial
import itertools

import networkx
import pandas
from pandas import DataFrame

from mlinspect.instrumentation.analyzers.analyzer_input import AnalyzerInputUnaryOperator, AnalyzerInputRow, \
    AnalyzerInputDataSource, OperatorContext
from mlinspect.instrumentation.backends.backend import Backend
from mlinspect.instrumentation.backends.pandas_backend_frame_wrapper import MlinspectDataFrame
from mlinspect.instrumentation.dag_node import OperatorType, DagNodeIdentifier


class PandasBackend(Backend):
    """
    The pandas backend
    """

    prefix = "pandas"

    operator_map = {
        ('pandas.io.parsers', 'read_csv'): OperatorType.DATA_SOURCE,
        ('pandas.core.frame', 'dropna'): OperatorType.SELECTION,
        ('pandas.core.frame', '__getitem__'): OperatorType.PROJECTION,
    }

    replacement_type_map = {
        'mlinspect.instrumentation.backends.pandas_backend_frame_wrapper': 'pandas.core.frame'
    }

    def preprocess_wir(self, wir: networkx.DiGraph) -> networkx.DiGraph:
        """
        Nothing to do here
        """
        return wir

    def postprocess_wir(self, wir: networkx.DiGraph) -> networkx.DiGraph:
        """
        Nothing to do here
        """
        return wir

    def __init__(self):
        super().__init__()
        self.input_data = None

    def before_call_used_value(self, function_info, subscript, call_code, value_code, value_value,
                               code_reference):
        """The value or module a function may be called on"""
        # pylint: disable=too-many-arguments
        if function_info == ('pandas.core.frame', 'dropna'):
            assert isinstance(value_value, MlinspectDataFrame)
            value_value['mlinspect_index'] = range(1, len(value_value) + 1)
            self.input_data = value_value
        elif function_info == ('pandas.core.frame', '__getitem__'):
            # TODO: Can this also be a select?
            assert isinstance(value_value, MlinspectDataFrame)
            value_value['mlinspect_index'] = range(1, len(value_value) + 1)
            self.input_data = value_value

    def before_call_used_args(self, function_info, subscript, call_code, args_code, code_reference, args_values):
        """The arguments a function may be called with"""
        # pylint: disable=too-many-arguments
        self.before_call_used_args_add_description(args_values, code_reference, function_info)

    def before_call_used_args_add_description(self, args_values, code_reference, function_info):
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
            self.code_reference_to_description[code_reference] = description

    def before_call_used_kwargs(self, function_info, subscript, call_code, kwargs_code, code_reference, kwargs_values):
        """The keyword arguments a function may be called with"""
        # pylint: disable=too-many-arguments, unused-argument, no-self-use, unnecessary-pass
        pass

    def after_call_used(self, function_info, subscript, call_code, return_value, code_reference):
        """The return value of some function"""
        # pylint: disable=too-many-arguments
        if function_info == ('pandas.io.parsers', 'read_csv'):
            return_value = self.execute_analyzer_visits_data_source(code_reference,
                                                                    return_value, function_info)
        elif function_info == ('pandas.core.frame', 'dropna'):
            operator_context = OperatorContext(OperatorType.SELECTION, function_info)
            return_value = self.execute_analyzer_visits_unary_operator(operator_context, code_reference, return_value,
                                                                       function_info)
        elif function_info == ('pandas.core.frame', '__getitem__'):
            # TODO: Can this also be a select
            operator_context = OperatorContext(OperatorType.PROJECTION, function_info)
            return_value['mlinspect_index'] = range(1, len(return_value) + 1)
            return_value = self.execute_analyzer_visits_unary_operator(operator_context, code_reference, return_value,
                                                                       function_info)

        self.input_data = None

        return return_value

    def execute_analyzer_visits_data_source(self, code_reference, return_value, function_info):
        """Execute analyzers when the current operator is a data source and does not have parents in the DAG"""
        operator_context = OperatorContext(OperatorType.DATA_SOURCE, function_info)
        annotation_iterators = []
        for analyzer in self.analyzers:
            iterator_for_analyzer = iter_input_data_source(return_value)
            annotation_iterator = analyzer.visit_operator(operator_context, iterator_for_analyzer)
            annotation_iterators.append(annotation_iterator)
        return_value = self.store_analyzer_outputs(annotation_iterators, code_reference, return_value, function_info)
        return return_value

    def store_analyzer_outputs(self, annotation_iterators, code_reference, return_value, function_info):
        """
        Stores the analyzer annotations for the rows in the dataframe and the
        analyzer annotations for the DAG operators in a map
        """
        annotation_iterators = itertools.zip_longest(*annotation_iterators)
        analyzer_names = [str(analyzer) for analyzer in self.analyzers]
        annotations_df = DataFrame(annotation_iterators, columns=analyzer_names)
        annotations_df['mlinspect_index'] = range(1, len(annotations_df) + 1)
        analyzer_outputs = {}
        for analyzer in self.analyzers:
            # TODO: Create arrays only once, return iterators over those same arrays repeatedly
            analyzer_output = analyzer.get_operator_annotation_after_visit()
            analyzer_outputs[analyzer] = analyzer_output
        dag_node_identifier = DagNodeIdentifier(self.operator_map[function_info], code_reference,
                                                self.code_reference_to_description.get(code_reference))
        self.dag_node_identifier_to_analyzer_output[dag_node_identifier] = analyzer_outputs
        return_value = MlinspectDataFrame(return_value)
        return_value.annotations = annotations_df
        self.input_data = None
        if "mlinspect_index" in return_value.columns:
            return_value = return_value.drop("mlinspect_index", axis=1)
        assert "mlinspect_index" not in return_value.columns
        assert isinstance(return_value, MlinspectDataFrame)
        return return_value

    def execute_analyzer_visits_unary_operator(self, operator_context, code_reference, return_value, function_info):
        """Execute analyzers when the current operator has one parent in the DAG"""
        assert "mlinspect_index" in return_value.columns
        assert isinstance(self.input_data, MlinspectDataFrame)
        annotation_iterators = []
        for analyzer in self.analyzers:
            analyzer_count = len(self.analyzers)
            analyzer_index = self.analyzers.index(analyzer)
            iterator_for_analyzer = iter_input_annotation_output(analyzer_count,
                                                                 analyzer_index,
                                                                 self.input_data,
                                                                 self.input_data.annotations,
                                                                 return_value)
            annotations_iterator = analyzer.visit_operator(operator_context, iterator_for_analyzer)
            annotation_iterators.append(annotations_iterator)
        return_value = self.store_analyzer_outputs(annotation_iterators, code_reference, return_value, function_info)
        return return_value


def iter_input_data_source(output):
    """
    Create an efficient iterator for the analyzer input for operators with no parent: Data Source
    """
    output = get_row_iterator(output)
    return map(AnalyzerInputDataSource, output)


def iter_input_annotation_output(analyzer_count, analyzer_index, input_data, input_annotations, output):
    """
    Create an efficient iterator for the analyzer input for operators with one parent.
    """
    # pylint: disable=too-many-locals
    # Performance tips:
    # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
    data_before_with_annotations = pandas.merge(input_data, input_annotations, left_on="mlinspect_index",
                                                right_on="mlinspect_index")
    joined_df = pandas.merge(data_before_with_annotations, output, left_on="mlinspect_index",
                             right_on="mlinspect_index")

    column_index_input_end = len(input_data.columns)
    column_annotation_current_analyzer = column_index_input_end + analyzer_index
    column_index_annotation_end = column_index_input_end + analyzer_count

    input_df_view = joined_df.iloc[:, 0:column_index_input_end - 1]
    input_df_view.columns = input_data.columns[0:-1]

    annotation_df_view = joined_df.iloc[:, column_annotation_current_analyzer:column_annotation_current_analyzer + 1]

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
    arrays.extend(dataframe.iloc[:, k] for k in range(0, len(dataframe.columns)))

    partial_func_create_row = partial(AnalyzerInputRow, fields=fields)
    test = map(partial_func_create_row, map(list, zip(*arrays)))
    return test
