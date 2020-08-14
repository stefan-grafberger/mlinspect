"""
A wrapper for sklearn transformers to capture method calls we do not see otherwise because of the pipeline
definition style
"""
import inspect
import itertools
from functools import partial

from pandas import DataFrame
from sklearn.base import BaseEstimator

from mlinspect.instrumentation.analyzers.analyzer_input import OperatorContext, AnalyzerInputRow, \
    AnalyzerInputUnaryOperator
from mlinspect.instrumentation.backends.pandas_backend_frame_wrapper import MlinspectDataFrame
from mlinspect.instrumentation.backends.sklearn_backend_ndarray_wrapper import MlinspectNdarray
from mlinspect.instrumentation.dag_node import CodeReference, OperatorType


class MlinspectEstimatorTransformer(BaseEstimator):
    """
    A wrapper for sklearn transformers to capture method calls we do not see otherwise because of the pipeline
    definition style
    See: https://scikit-learn.org/stable/developers/develop.html
    """

    def __init__(self, transformer, code_reference: CodeReference, analyzers, code_reference_analyzer_output_map):
        self.transformer = transformer
        self.name = transformer.__class__.__name__

        module = inspect.getmodule(transformer)
        self.module_name = module.__name__
        self.call_function_info = (module.__name__, transformer.__class__.__name__)
        self.code_reference = code_reference
        self.analyzers = analyzers
        self.code_reference_analyzer_output_map = code_reference_analyzer_output_map

    def fit(self, X: list, y=None) -> 'MlinspectEstimatorTransformer':
        """
        Override fit
        """
        # FIXME: move this somewhere else
        #  associate this with the DAG node

        # pylint: disable=invalid-name
        if self.call_function_info == ('sklearn.pipeline', 'Pipeline'):
            # Pipeline.fit returns nothing (trained models get no edge in our DAG)
            # Only need to do two scans for train data and train labels
            # train data:

            # TODO: returns nothing but allows a scan for train data and train labels. output is same as input
            function_info = (self.module_name, "fit")  # TODO: nested pipelines
            operator_context = OperatorContext(OperatorType.TRAIN_DATA, function_info)
            execute_analyzer_visits_df_input_df_output(operator_context, self.code_reference, X, X, self.analyzers,
                                                       self.code_reference_analyzer_output_map)
            pass
        print(self.call_function_info[1])
        print("fit:")
        print("X")
        print(X)
        print("y")
        print(y)
        self.transformer = self.transformer.fit(X, y)
        print("analyzer output:")
        print(self.code_reference_analyzer_output_map)
        return self

    def transform(self, X: list) -> list:
        """
        Override transform
        """
        # pylint: disable=invalid-name
        print(self.call_function_info[1])
        print("transform:")
        print("X")
        print(X)
        result = self.transformer.transform(X)
        print("result")
        print(result)
        return result

    def fit_transform(self, X: list, y=None) -> list:  # TODO: There can be some additional kwargs sometimes
        """
        Override fit_transform
        """
        # pylint: disable=invalid-name
        if self.call_function_info == ('sklearn.compose._column_transformer', 'ColumnTransformer'):
            print("column transformer")

        print(self.call_function_info[1])
        print("fit_transform:")
        print("X")
        print(X)
        print("y")
        print(y)
        result = self.transformer.fit_transform(X, y)
        print("result")
        print(result)
        return result


def execute_analyzer_visits_df_input_df_output(operator_context, code_reference, input_data, output_data, analyzers,
                                               code_reference_analyzer_output_map):
    """Execute analyzers when the current operator has one parent in the DAG"""
    # assert "mlinspect_index" in output_data.columns
    assert isinstance(input_data, MlinspectDataFrame)
    annotation_iterators = []
    for analyzer in analyzers:
        analyzer_count = len(analyzers)
        analyzer_index = analyzers.index(analyzer)
        iterator_for_analyzer = iter_input_annotation_output_df_df(analyzer_index,
                                                                   input_data,
                                                                   input_data.annotations,
                                                                   output_data)
        annotations_iterator = analyzer.visit_operator(operator_context, iterator_for_analyzer)
        annotation_iterators.append(annotations_iterator)
    return_value = store_analyzer_outputs(annotation_iterators, code_reference, output_data, analyzers,
                                          code_reference_analyzer_output_map)
    return return_value


def iter_input_annotation_output_df_df(analyzer_index, input_df, annotation_df, output_df):
    """
    Create an efficient iterator for the analyzer input for operators with one parent.
    """
    # pylint: disable=too-many-locals
    # Performance tips:
    # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas

    annotation_df_view = annotation_df.iloc[:, analyzer_index:analyzer_index + 1]

    input_rows = get_df_row_iterator(input_df)
    annotation_rows = get_df_row_iterator(annotation_df_view)
    output_rows = get_df_row_iterator(output_df)

    return map(lambda input_tuple: AnalyzerInputUnaryOperator(*input_tuple),
               zip(input_rows, annotation_rows, output_rows))


def get_df_row_iterator(dataframe):
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


def store_analyzer_outputs(annotation_iterators, code_reference, return_value, analyzers,
                           code_reference_analyzer_output_map):
    """
    Stores the analyzer annotations for the rows in the dataframe and the
    analyzer annotations for the DAG operators in a map
    """
    annotation_iterators = itertools.zip_longest(*annotation_iterators)
    analyzer_names = [str(analyzer) for analyzer in analyzers]
    annotations_df = DataFrame(annotation_iterators, columns=analyzer_names)
    analyzer_outputs = {}
    for analyzer in analyzers:
        analyzer_output = analyzer.get_operator_annotation_after_visit()
        analyzer_outputs[analyzer] = analyzer_output
    # FIXME: code_reference_analyzer_output_map[code_reference] = analyzer_outputs
    # FIXME: Use the code_reference here. Then have a special mechanism for post-processing the wir with the
    #  sklearn backend. We can save in a map the analyzers with one or multiple code references.
    #  for  post processing the wir visit all nodes that are sklearn nodes. than lookup this map.
    return_value = MlinspectNdarray(return_value) # save with nesting level and order to identify transformers
    return_value.annotations = annotations_df
    # self.input_data = None
    assert isinstance(return_value, MlinspectNdarray)
    return return_value
