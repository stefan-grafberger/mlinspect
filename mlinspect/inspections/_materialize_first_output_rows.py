"""
A simple inspection to materialise operator outputs
"""
from typing import Iterable

from pandas import DataFrame

from ._inspection import Inspection
from ._inspection_input import InspectionInputSinkOperator, OperatorType


class MaterializeFirstOutputRows(Inspection):
    """
    A simple example analyzer
    """

    def __init__(self, row_count: int):
        self.row_count = row_count
        self._analyzer_id = self.row_count
        self._first_rows_op_output = None
        self._operator_type = None
        self._output_columns = None

    @property
    def inspection_id(self):
        return self._analyzer_id

    def visit_operator(self, inspection_input) -> Iterable[any]:
        """
        Visit an operator
        """
        current_count = - 1
        operator_output = []
        self._operator_type = inspection_input.operator_context.operator

        if not isinstance(inspection_input, InspectionInputSinkOperator):
            self._output_columns = inspection_input.output_columns.fields
            for row in inspection_input.row_iterator:
                current_count += 1
                if current_count < self.row_count:
                    operator_output.append(row.output)
                yield None
        else:
            for _ in inspection_input.row_iterator:
                yield None

        self._first_rows_op_output = operator_output

    def get_operator_annotation_after_visit(self) -> any:
        assert self._operator_type
        if self._operator_type is not OperatorType.ESTIMATOR:
            assert self._first_rows_op_output and self._output_columns is not None  # Visit must be finished
            result = DataFrame(self._first_rows_op_output, columns=self._output_columns)
            self._first_rows_op_output = None
            self._operator_type = None
            self._output_columns = None
            return result
        self._operator_type = None
        self._output_columns = None
        return None
