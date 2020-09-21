"""
Some utility functions the different instrumentation backends
"""
import itertools
from functools import partial

import numpy
from pandas import DataFrame, Series
from scipy.sparse import csr_matrix

from ..inspections.inspection_input import InspectionInputRow


def build_annotation_df_from_iters(inspections, annotation_iterators):
    """
    Build the annotations dataframe
    """
    annotation_iterators = itertools.zip_longest(*annotation_iterators)
    inspection_names = [str(inspection) for inspection in inspections]
    annotations_df = DataFrame(annotation_iterators, columns=inspection_names)
    return annotations_df


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
    else:
        assert False
    return iterator


def get_df_row_iterator(dataframe):
    """
    Create an efficient iterator for the data frame rows.
    The implementation is inspired by the implementation of the pandas DataFrame.itertuple method
    """
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
