"""
A simple example analyzer
"""
from typing import Union, Iterable

from .inspection import Inspection
from .inspection_input import OperatorContext, InspectionInputDataSource, \
    InspectionInputUnaryOperator
from ..instrumentation.dag_node import OperatorType


class MaterializeFirstRowsInspection(Inspection):
    """
    A simple example analyzer
    """

    def __init__(self, row_count: int):
        self.row_count = row_count
        self._analyzer_id = self.row_count
        self._first_rows_op_output = None
        self._operator_type = None

    @property
    def inspection_id(self):
        return self._analyzer_id

    def visit_operator(self, operator_context: OperatorContext,
                       row_iterator: Union[Iterable[InspectionInputDataSource], Iterable[InspectionInputUnaryOperator]])\
            -> Iterable[any]:
        """
        Visit an operator
        """
        current_count = - 1
        operator_output = []
        self._operator_type = operator_context.operator

        if self._operator_type is not OperatorType.ESTIMATOR:
            for row in row_iterator:
                current_count += 1
                if current_count < self.row_count:
                    operator_output.append(row.output)
                yield None
        else:
            for _ in row_iterator:
                yield None

        self._first_rows_op_output = operator_output

    def get_operator_annotation_after_visit(self) -> any:
        assert self._operator_type
        if self._operator_type is not OperatorType.ESTIMATOR:
            assert self._first_rows_op_output  # May only be called after the operator visit is finished
            result = self._first_rows_op_output
            self._first_rows_op_output = None
            self._operator_type = None
            return result
        self._operator_type = None
        return None
