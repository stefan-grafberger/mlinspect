"""
Some utility functions the different instrumentation backends
"""
import itertools
from functools import partial

import numpy
from pandas import DataFrame, Series
from scipy.sparse import csr_matrix

from .pandas_backend_frame_wrapper import MlinspectDataFrame, MlinspectSeries
from .sklearn_backend_csr_matrx_wrapper import MlinspectCsrMatrix
from .sklearn_backend_ndarray_wrapper import MlinspectNdarray
from ..inspections.inspection_input import InspectionInputRow


def get_annotation_rows(input_annotations, inspection_index):
    """
    In the pandas backend, we store annotations in a data frame, for the sklearn transformers lists are enough
    """
    if isinstance(input_annotations, DataFrame):
        annotation_df_view = input_annotations.iloc[:, inspection_index]
    else:
        annotation_df_view = input_annotations[inspection_index]
    annotation_rows = get_iterator_for_type(annotation_df_view, True)
    return annotation_rows


def build_annotation_df_from_iters(inspections, annotation_iterators):
    """
    Build the annotations dataframe
    """
    annotation_iterators = itertools.zip_longest(*annotation_iterators)
    inspection_names = [str(inspection) for inspection in inspections]
    annotations_df = DataFrame(annotation_iterators, columns=inspection_names)
    return annotations_df


def build_annotation_list_from_iters(annotation_iterators):
    """
    Build the annotations dataframe
    """
    annotation_lists = [list(iterator) for iterator in annotation_iterators]
    return list(annotation_lists)


def get_iterator_for_type(data, np_nditer_with_refs=False):
    """
    Create an efficient iterator for the data.
    Automatically detects the data type and fails if it cannot handle that data type.
    """
    if isinstance(data, DataFrame):
        iterator = get_df_row_iterator(data)
    elif isinstance(data, numpy.ndarray):
        # TODO: Measure performance impact of np_nditer_with_refs. To support arbitrary pipelines, remove this
        #  or check the type of the standard iterator. It seems the nditer variant is faster but does not always work
        iterator = get_numpy_array_row_iterator(data, np_nditer_with_refs)
    elif isinstance(data, Series):
        iterator = get_series_row_iterator(data)
    elif isinstance(data, csr_matrix):
        iterator = get_csr_row_iterator(data)
    elif isinstance(data, list):
        iterator = get_list_row_iterator(data)
    else:
        assert False
    return iterator


def create_wrapper_with_annotations(annotations_df, return_value, pandas_backend=None):
    """
    Create a wrapper based on the data type of the return value and store the annotations in it.
    """
    if isinstance(return_value, numpy.ndarray):
        return_value = MlinspectNdarray(return_value)
        return_value.annotations = annotations_df
        new_return_value = return_value
    elif isinstance(return_value, (DataFrame, MlinspectDataFrame)):
        if not pandas_backend:
            pandas_backend = return_value.backend
        return_value = MlinspectDataFrame(return_value)
        return_value.annotations = annotations_df

        assert pandas_backend  # This is needed to deal with ops like adding new columns
        return_value.backend = pandas_backend

        # Remove index columns that may have been created
        if "mlinspect_index" in return_value.columns:
            return_value = return_value.drop("mlinspect_index", axis=1)
        elif "mlinspect_index_x" in return_value.columns:
            return_value = return_value.drop(["mlinspect_index_x", "mlinspect_index_y"], axis=1)
        assert "mlinspect_index" not in return_value.columns
        assert "mlinspect_index_x" not in return_value.columns

        new_return_value = return_value
    elif isinstance(return_value, Series):
        return_value = MlinspectSeries(return_value)
        return_value.annotations = annotations_df
        new_return_value = return_value
    elif isinstance(return_value, csr_matrix):
        return_value = MlinspectCsrMatrix(return_value)
        return_value.annotations = annotations_df
        new_return_value = return_value
    else:
        assert False
    return new_return_value


def get_df_row_iterator(dataframe):
    """
    Create an efficient iterator for the data frame rows.
    The implementation is inspired by the implementation of the pandas DataFrame.itertuple method
    """
    # Performance tips:
    # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
    arrays = []
    fields = list(dataframe.columns)
    arrays.extend(dataframe.iloc[:, k] for k in range(0, len(dataframe.columns)))

    partial_func_create_row = partial(InspectionInputRow, fields=fields)
    return map(partial_func_create_row, map(list, zip(*arrays)))


def get_series_row_iterator(series):
    """
    Create an efficient iterator for the data frame rows.
    The implementation is inspired by the implementation of the pandas DataFrame.itertuple method
    """
    fields = list(["array"])
    numpy_iterator = series.__iter__()
    partial_func_create_row = partial(InspectionInputRow, fields=fields)

    return map(partial_func_create_row, map(list, zip(numpy_iterator)))


def get_numpy_array_row_iterator(nparray, nditer=False):
    """
    Create an efficient iterator for the data frame rows.
    The implementation is inspired by the implementation of the pandas DataFrame.itertuple method
    """
    fields = list(["array"])
    if nditer is True:
        numpy_iterator = numpy.nditer(nparray, ["refs_ok"])
    else:
        numpy_iterator = nparray.__iter__()
    partial_func_create_row = partial(InspectionInputRow, fields=fields)

    return map(partial_func_create_row, map(list, zip(numpy_iterator)))


def get_list_row_iterator(list_data):
    """
    Create an efficient iterator for the data frame rows.
    The implementation is inspired by the implementation of the pandas DataFrame.itertuple method
    """
    fields = list(["array"])
    numpy_iterator = list_data.__iter__()
    partial_func_create_row = partial(InspectionInputRow, fields=fields)

    return map(partial_func_create_row, map(list, zip(numpy_iterator)))


def get_csr_row_iterator(csr):
    """
    Create an efficient iterator for csr rows.
    The implementation is inspired by the implementation of the pandas DataFrame.itertuple method
    """
    # TODO: Maybe there is a way to use sparse rows that is faster
    #  However, this is the fastest way I discovered so far
    np_array = csr.toarray()
    fields = list(["array"])
    numpy_iterator = np_array.__iter__()
    partial_func_create_row = partial(InspectionInputRow, fields=fields)

    return map(partial_func_create_row, map(list, zip(numpy_iterator)))
