"""
Some utility functions the different instrumentation backends
"""
import itertools

import numpy
from pandas import DataFrame, Series
from scipy.sparse import csr_matrix

from ._backend import AnnotatedDfObject
from ..inspections._inspection_input import ColumnInfo
from ..monkeypatching._mlinspect_ndarray import MlinspectNdarray


def get_annotation_rows(input_annotations, inspection_index):
    """
    In the pandas backend, we store annotations in a data frame, for the sklearn transformers lists are enough
    """
    if isinstance(input_annotations, DataFrame):
        annotations_for_inspection = input_annotations.iloc[:, inspection_index]
        assert isinstance(annotations_for_inspection, Series)
    else:
        annotations_for_inspection = input_annotations[inspection_index]
        assert isinstance(annotations_for_inspection, list)
    annotation_rows = annotations_for_inspection.__iter__()
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


def get_iterator_for_type(data, np_nditer_with_refs=False, columns=None):
    """
    Create an efficient iterator for the data.
    Automatically detects the data type and fails if it cannot handle that data type.
    """
    if isinstance(data, DataFrame):
        iterator = get_df_row_iterator(data)
    elif isinstance(data, numpy.ndarray):
        # TODO: Measure performance impact of np_nditer_with_refs. To support arbitrary pipelines, remove this
        #  or check the type of the standard iterator. It seems the nditer variant is faster but does not always work
        iterator = get_numpy_array_row_iterator(data, np_nditer_with_refs, columns)
    elif isinstance(data, Series):
        iterator = get_series_row_iterator(data, columns)
    elif isinstance(data, csr_matrix):
        iterator = get_csr_row_iterator(data, columns)
    elif isinstance(data, list):
        iterator = get_list_row_iterator(data, columns)
    else:
        raise NotImplementedError("TODO: Support type {}!".format(type(data)))
    return iterator


def create_wrapper_with_annotations(annotations_df, return_value) -> AnnotatedDfObject:
    """
    Create a wrapper based on the data type of the return value and store the annotations in it.
    """
    if isinstance(return_value, numpy.ndarray):
        return_value = MlinspectNdarray(return_value)
        new_return_value = AnnotatedDfObject(return_value, annotations_df)
    elif isinstance(return_value, DataFrame):
        # Remove index columns that may have been created
        if "mlinspect_index" in return_value.columns:
            return_value = return_value.drop("mlinspect_index", axis=1)
        elif "mlinspect_index_x" in return_value.columns:
            return_value = return_value.drop(["mlinspect_index_x", "mlinspect_index_y"], axis=1)
        assert "mlinspect_index" not in return_value.columns
        assert "mlinspect_index_x" not in return_value.columns

        new_return_value = AnnotatedDfObject(return_value, annotations_df)
    elif isinstance(return_value, (Series, csr_matrix)):
        return_value.annotations = annotations_df
        new_return_value = AnnotatedDfObject(return_value, annotations_df)
    elif return_value is None:
        new_return_value = AnnotatedDfObject(None, annotations_df)
    else:
        raise NotImplementedError("A type that is still unsupported was found: {}".format(return_value))
    return new_return_value


def get_df_row_iterator(dataframe):
    """
    Create an efficient iterator for the data frame rows.
    The implementation is inspired by the implementation of the pandas DataFrame.itertuple method
    """
    # Performance tips:
    # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
    arrays = []
    column_info = ColumnInfo(list(dataframe.columns.values))
    arrays.extend(dataframe.iloc[:, k] for k in range(0, len(dataframe.columns)))

    return column_info, map(tuple, zip(*arrays))


def get_series_row_iterator(series, columns=None):
    """
    Create an efficient iterator for the data frame rows.
    The implementation is inspired by the implementation of the pandas DataFrame.itertuple method
    """
    if columns:
        column_info = ColumnInfo(columns)
    elif series.name:
        column_info = ColumnInfo([series.name])
    else:
        column_info = ColumnInfo(["array"])
    numpy_iterator = series.__iter__()

    return column_info, map(tuple, zip(numpy_iterator))


def get_numpy_array_row_iterator(nparray, nditer=False, columns=None):
    """
    Create an efficient iterator for the data frame rows.
    The implementation is inspired by the implementation of the pandas DataFrame.itertuple method
    """
    if columns:
        column_info = ColumnInfo(columns)
    else:
        column_info = ColumnInfo(["array"])
    if nditer is True:
        numpy_iterator = numpy.nditer(nparray, ["refs_ok"])
    else:
        numpy_iterator = nparray.__iter__()

    return column_info, map(tuple, zip(numpy_iterator))


def get_list_row_iterator(list_data, columns=None):
    """
    Create an efficient iterator for the data frame rows.
    The implementation is inspired by the implementation of the pandas DataFrame.itertuple method
    """
    if columns:
        column_info = ColumnInfo(columns)
    else:
        column_info = ColumnInfo(["array"])
    numpy_iterator = list_data.__iter__()

    return column_info, map(tuple, zip(numpy_iterator))


def get_csr_row_iterator(csr, columns=None):
    """
    Create an efficient iterator for csr rows.
    The implementation is inspired by the implementation of the pandas DataFrame.itertuple method
    """
    # TODO: Maybe there is a way to use sparse rows that is faster
    #  However, this is the fastest way I discovered so far
    np_array = csr.toarray()
    if columns:
        column_info = ColumnInfo(columns)
    else:
        column_info = ColumnInfo(["array"])
    numpy_iterator = np_array.__iter__()

    return column_info, map(tuple, zip(numpy_iterator))
