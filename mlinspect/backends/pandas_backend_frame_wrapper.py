"""
A wrapper for pandas dataframes and series to store our additional annotations
"""

from pandas import DataFrame, Series


class MlinspectSeries(Series):
    """
    A Series wrapper also storing annotations.
    See the pandas documentation: https://pandas.pydata.org/pandas-docs/stable/development/extending.html
    """
    # pylint: disable=too-many-ancestors

    _metadata = ['annotations', 'backend']

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

    _metadata = ['annotations', 'backend']

    @property
    def _constructor(self):
        return MlinspectDataFrame

    @property
    def _constructor_sliced(self):
        return MlinspectSeries

    @property
    def _constructor_expanddim(self):
        return MlinspectDataFrame

    def __setitem__(self, key, value):
        if key not in {'mlinspect_index', 'mlinspect_index_x', 'mlinspect_index_y'}:
            assert self.backend

            previous_df = self.copy()
            super()._set_item(key, value)
            self.backend.after_call_used_setkey(key, previous_df, self)
        else:
            super()._set_item(key, value)
