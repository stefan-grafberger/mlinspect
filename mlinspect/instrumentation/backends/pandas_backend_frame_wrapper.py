"""
A wrapper for pandas dataframes and series to store our additional annotations
"""

from pandas import DataFrame, Series
from pandas.core.groupby import DataFrameGroupBy


class MlinspectSeries(Series):
    """
    A Series wrapper also storing annotations.
    See the pandas documentation: https://pandas.pydata.org/pandas-docs/stable/development/extending.html
    """
    # pylint: disable=too-many-ancestors

    _metadata = ['annotations']

    @property
    def _constructor(self):
        return MlinspectSeries

    @property
    def _constructor_expanddim(self):
        return MlinspectDataFrame


class MlinspectDataFrame(DataFrame):
    """
    A DataFrame wrapper also storing annotations.
    See the pandas documentation: https://pandas.pydata.org/pandas-docs/stable/development/extending.html
    """

    _metadata = ['annotations']

    @property
    def _constructor(self):
        return MlinspectDataFrame

    @property
    def _constructor_sliced(self):
        return MlinspectSeries

    @property
    def _constructor_expanddim(self):
        return MlinspectDataFrame
