"""
Some utility functions the different instrumentation backends
"""
import itertools
from functools import partial

import numpy
from pandas import DataFrame

from mlinspect.instrumentation.analyzers.analyzer_input import AnalyzerInputRow


def build_annotation_df_from_iters(analyzers, annotation_iterators):
    """
    Build the annotations dataframe
    """
    annotation_iterators = itertools.zip_longest(*annotation_iterators)
    analyzer_names = [str(analyzer) for analyzer in analyzers]
    annotations_df = DataFrame(annotation_iterators, columns=analyzer_names)
    return annotations_df


def get_df_row_iterator(dataframe):
    """
    Create an efficient iterator for the data frame rows.
    The implementation is inspired by the implementation of the pandas DataFrame.itertuple method
    """
    arrays = []
    fields = list(dataframe.columns)
    arrays.extend(dataframe.iloc[:, k] for k in range(0, len(dataframe.columns)))

    partial_func_create_row = partial(AnalyzerInputRow, fields=fields)
    return map(partial_func_create_row, map(list, zip(*arrays)))


def get_series_row_iterator(series):
    """
    Create an efficient iterator for the data frame rows.
    The implementation is inspired by the implementation of the pandas DataFrame.itertuple method
    """
    arrays = []
    fields = list(["array"])
    numpy_iterator = series.__iter__()
    partial_func_create_row = partial(AnalyzerInputRow, fields=fields)

    return map(partial_func_create_row, map(list, zip(numpy_iterator)))


def get_numpy_array_row_iterator(nparray):
    """
    Create an efficient iterator for the data frame rows.
    The implementation is inspired by the implementation of the pandas DataFrame.itertuple method
    """
    fields = list(["array"])
    numpy_iterator = numpy.nditer(nparray, ["refs_ok"])
    partial_func_create_row = partial(AnalyzerInputRow, fields=fields)

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
    partial_func_create_row = partial(AnalyzerInputRow, fields=fields)

    return map(partial_func_create_row, map(list, zip(numpy_iterator)))
