"""
A inspection to compute the ratio of non-values in output columns
"""
from typing import Iterable

import pandas

from mlinspect.inspections._inspection import Inspection
from mlinspect.inspections._inspection_input import OperatorType, InspectionInputSinkOperator


class CompletenessOfColumns(Inspection):
    """
    An inspection to compute the completeness of columns
    """

    def __init__(self, columns):
        self._present_column_names = []
        self._null_value_counts = []
        self._total_counts = []
        self._operator_type = None
        self.columns = columns

    @property
    def inspection_id(self):
        return tuple(self.columns)

    def visit_operator(self, inspection_input) -> Iterable[any]:
        """
        Visit an operator
        """
        # pylint: disable=too-many-branches, too-many-statements, too-many-locals
        self._present_column_names = []
        self._null_value_counts = []
        self._total_counts = []
        self._operator_type = inspection_input.operator_context.operator

        if not isinstance(inspection_input, InspectionInputSinkOperator):
            present_columns_index = []
            for column_name in self.columns:
                column_present = column_name in inspection_input.output_columns.fields
                if column_present:
                    column_index = inspection_input.output_columns.get_index_of_column(column_name)
                    present_columns_index.append(column_index)
                    self._present_column_names.append(column_name)
                    self._null_value_counts.append(0)
                    self._total_counts.append(0)
            for row in inspection_input.row_iterator:
                for present_column_index, column_index in enumerate(present_columns_index):
                    column_value = row.output[column_index]
                    is_null = pandas.isna(column_value)
                    self._null_value_counts[present_column_index] += int(is_null)
                    self._total_counts[present_column_index] += 1
                yield None
        else:
            for _ in inspection_input.row_iterator:
                yield None

    def get_operator_annotation_after_visit(self) -> any:
        assert self._operator_type
        if self._operator_type is not OperatorType.ESTIMATOR:
            completeness_results = {}
            for column_index, column_name in enumerate(self._present_column_names):
                null_value_count = self._null_value_counts[column_index]
                total_count = self._total_counts[column_index]
                completeness = (total_count - null_value_count) / total_count
                completeness_results[column_name] = completeness
            return completeness_results
        self._operator_type = None
        return None
