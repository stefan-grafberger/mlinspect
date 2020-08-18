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
        if key != 'mlinspect_index':
            print("hello world")
        #if self.backend and key != 'mlinspect_index':
        #    self.backend.before_call_index_assign(self, key, value)
        super()._set_item(key, value)
        #if self.backend and key != 'mlinspect_index':
        #    self.backend.after_call_index_assign(self, key, value)
        # TODO: Add backend methods and a DAG postprocessor that adds this module info.
