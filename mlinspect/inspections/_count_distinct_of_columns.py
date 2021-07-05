"""
An inspection to compute the number of distinct values in output columns
"""
from typing import Iterable

from mlinspect.inspections._inspection import Inspection
from mlinspect.inspections._inspection_input import OperatorType, InspectionInputSinkOperator


class CountDistinctOfColumns(Inspection):
    """
    An inspection to compute the number of distinct values of columns
    """

    def __init__(self, columns):
        self._present_column_names = []
        self._distinct_value_sets = []
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
        self._distinct_value_sets = []
        self._operator_type = inspection_input.operator_context.operator

        if not isinstance(inspection_input, InspectionInputSinkOperator):
            present_columns_index = []
            for column_name in self.columns:
                column_present = column_name in inspection_input.output_columns.fields
                if column_present:
                    column_index = inspection_input.output_columns.get_index_of_column(column_name)
                    present_columns_index.append(column_index)
                    self._present_column_names.append(column_name)
                    self._distinct_value_sets.append(set())
            for row in inspection_input.row_iterator:
                for present_column_index, column_index in enumerate(present_columns_index):
                    column_value = row.output[column_index]
                    self._distinct_value_sets[present_column_index].add(column_value)
                yield None
        else:
            for _ in inspection_input.row_iterator:
                yield None

    def get_operator_annotation_after_visit(self) -> any:
        assert self._operator_type
        if self._operator_type is not OperatorType.ESTIMATOR:
            completeness_results = {}
            for column_index, column_name in enumerate(self._present_column_names):
                distinct_value_count = len(self._distinct_value_sets[column_index])
                completeness_results[column_name] = distinct_value_count
            del self._distinct_value_sets
            return completeness_results
        self._operator_type = None
        return None
