"""
The scikit-learn backend
"""
import itertools
from functools import partial

import networkx
import numpy
from pandas import DataFrame

from mlinspect.instrumentation.analyzers.analyzer_input import OperatorContext, AnalyzerInputRow, \
    AnalyzerInputUnaryOperator
from mlinspect.instrumentation.backends.backend import Backend
from mlinspect.instrumentation.backends.pandas_backend_frame_wrapper import MlinspectDataFrame
from mlinspect.instrumentation.backends.sklearn_backend_ndarray_wrapper import MlinspectNdarray
from mlinspect.instrumentation.backends.sklearn_wir_preprocessor import SklearnWirPreprocessor
from mlinspect.instrumentation.dag_node import OperatorType


class SklearnBackend(Backend):
    """
    The scikit-learn backend
    """

    prefix = "sklearn"

    operator_map = {
        ('sklearn.preprocessing._label', 'label_binarize'): OperatorType.PROJECTION_MODIFY,
        ('sklearn.compose._column_transformer', 'ColumnTransformer', 'Projection'): OperatorType.PROJECTION,
        ('sklearn.preprocessing._encoders', 'OneHotEncoder', 'Pipeline'): OperatorType.TRANSFORMER,
        ('sklearn.preprocessing._data', 'StandardScaler', 'Pipeline'): OperatorType.TRANSFORMER,
        ('sklearn.compose._column_transformer', 'ColumnTransformer', 'Concatenation'): OperatorType.CONCATENATION,
        ('sklearn.tree._classes', 'DecisionTreeClassifier', 'Pipeline'): OperatorType.ESTIMATOR,
        ('sklearn.pipeline', 'fit', 'Pipeline'): OperatorType.FIT,
        ('sklearn.pipeline', 'fit', 'Train Data'): OperatorType.TRAIN_DATA,
        ('sklearn.pipeline', 'fit', 'Train Labels'): OperatorType.TRAIN_LABELS
    }

    replacement_type_map = {}

    @staticmethod
    def preprocess_wir(wir: networkx.DiGraph) -> networkx.DiGraph:
        """
        Preprocess scikit-learn pipeline operations to hide the special pipeline
        declaration style from other parts of the library
        """
        return SklearnWirPreprocessor().preprocess_wir(wir)

    def __init__(self):
        super().__init__()
        self.input_data = None

    def before_call_used_value(self, function_info, subscript, call_code, value_code, value_value,
                               code_reference):
        """The value or module a function may be called on"""
        # pylint: disable=too-many-arguments, unused-argument, no-self-use
        pass

    def before_call_used_args(self, function_info, subscript, call_code, args_code, code_reference, args_values):
        """The arguments a function may be called with"""
        # pylint: disable=too-many-arguments, unused-argument, no-self-use
        description = None

        if function_info == ('sklearn.preprocessing._encoders', 'OneHotEncoder'):
            description = "Categorical Encoder (OneHotEncoder)"
        elif function_info == ('sklearn.preprocessing._data', 'StandardScaler'):
            description = "Numerical Encoder (StandardScaler)"
        elif function_info == ('sklearn.tree._classes', 'DecisionTreeClassifier'):
            description = "Decision Tree"
        elif function_info == ('sklearn.preprocessing._label', 'label_binarize'):
            assert isinstance(args_values[0], MlinspectDataFrame)
            self.input_data = args_values[0]

        if description:
            self.code_reference_to_description[code_reference] = description

    def before_call_used_kwargs(self, function_info, subscript, call_code, kwargs_code, code_reference, kwargs_values):
        """The keyword arguments a function may be called with"""
        # pylint: disable=too-many-arguments, unused-argument, no-self-use
        description = None
        if function_info == ('sklearn.preprocessing._label', 'label_binarize'):
            classes = kwargs_values['classes']
            description = "label_binarize, classes: {}".format(classes)

        if description:
            self.code_reference_to_description[code_reference] = description

    def after_call_used(self, function_info, subscript, call_code, return_value, code_reference):
        """The return value of some function"""
        # pylint: disable=too-many-arguments, unused-argument, no-self-use
        if function_info == ('sklearn.preprocessing._label', 'label_binarize'):
            operator_context = OperatorContext(OperatorType.PROJECTION_MODIFY, function_info)
            self.execute_analyzer_visits_df_input_np_output(operator_context, code_reference, return_value)

        self.input_data = None

        return return_value

    def execute_analyzer_visits_df_input_np_output(self, operator_context, code_reference, return_value):
        """Execute analyzers when the current operator has one parent in the DAG"""
        assert isinstance(self.input_data, MlinspectDataFrame)
        annotation_iterators = []
        for analyzer in self.analyzers:
            analyzer_index = self.analyzers.index(analyzer)
            # TODO: Create arrays only once, return iterators over those same arrays repeatedly
            iterator_for_analyzer = iter_input_annotation_output(analyzer_index,
                                                                 self.input_data,
                                                                 self.input_data.annotations,
                                                                 return_value)
            annotations_iterator = analyzer.visit_operator(operator_context, iterator_for_analyzer)
            annotation_iterators.append(annotations_iterator)
        return_value = self.store_analyzer_outputs(annotation_iterators, code_reference, return_value)
        assert isinstance(return_value, MlinspectNdarray)
        return return_value

    def store_analyzer_outputs(self, annotation_iterators, code_reference, return_value):
        """
        Stores the analyzer annotations for the rows in the dataframe and the
        analyzer annotations for the DAG operators in a map
        """
        annotation_iterators = itertools.zip_longest(*annotation_iterators)
        analyzer_names = [str(analyzer) for analyzer in self.analyzers]
        annotations_df = DataFrame(annotation_iterators, columns=analyzer_names)
        analyzer_outputs = {}
        for analyzer in self.analyzers:
            analyzer_output = analyzer.get_operator_annotation_after_visit()
            analyzer_outputs[analyzer] = analyzer_output
        self.code_reference_analyzer_output_map[code_reference] = analyzer_outputs
        return_value = MlinspectNdarray(return_value)
        return_value.annotations = annotations_df
        self.input_data = None
        assert isinstance(return_value, MlinspectNdarray)
        return return_value


def iter_input_annotation_output(analyzer_index, input_df, annotation_df, output_array):
    """
    Create an efficient iterator for the analyzer input for operators with one parent.
    """
    # pylint: disable=too-many-locals
    # Performance tips:
    # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas

    annotation_df_view = annotation_df.iloc[:, analyzer_index:analyzer_index + 1]

    input_rows = get_df_row_iterator(input_df)
    annotation_rows = get_df_row_iterator(annotation_df_view)
    output_rows = get_numpy_array_row_iterator(output_array)

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


def get_numpy_array_row_iterator(nparray):
    """
    Create an efficient iterator for the data frame rows.
    The implementation is inspired by the implementation of the pandas DataFrame.itertuple method
    """
    fields = list(["array"])
    numpy_iterator = numpy.nditer(nparray, ["refs_ok"])
    partial_func_create_row = partial(AnalyzerInputRow, fields=fields)

    test = map(partial_func_create_row, zip(numpy_iterator))
    return test
