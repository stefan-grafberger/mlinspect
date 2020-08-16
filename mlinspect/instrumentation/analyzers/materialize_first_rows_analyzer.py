"""
A simple example analyzer
"""
from typing import Union, Iterable

from mlinspect.instrumentation.analyzers.analyzer_input import OperatorContext, AnalyzerInputDataSource, \
    AnalyzerInputUnaryOperator
from mlinspect.instrumentation.analyzers.analyzer import Analyzer
from mlinspect.instrumentation.dag_node import OperatorType


class MaterializeFirstRowsAnalyzer(Analyzer):
    """
    A simple example analyzer
    """

    def __init__(self, row_count: int):
        self.row_count = row_count
        self._analyzer_id = self.row_count
        self._operator_output = None
        self.operator_type = None

    @property
    def analyzer_id(self):
        return self._analyzer_id

    def visit_operator(self, operator_context: OperatorContext,
                       row_iterator: Union[Iterable[AnalyzerInputDataSource], Iterable[AnalyzerInputUnaryOperator]])\
            -> Iterable[any]:
        """
        Visit an operator
        """
        current_count = - 1
        operator_output = []
        self.operator_type = operator_context.operator

        if self.operator_type is not OperatorType.ESTIMATOR:
            for row in row_iterator:
                current_count += 1
                if current_count < self.row_count:
                    operator_output.append(row.output)
                yield None
        else:
            for _ in row_iterator:
                yield None

        self._operator_output = operator_output

    def get_operator_annotation_after_visit(self) -> any:
        assert self.operator_type
        if self.operator_type is not OperatorType.ESTIMATOR:
            assert self._operator_output  # May only be called after the operator visit is finished
            result = self._operator_output
            self._operator_output = None
            self.operator_type = None
            return result
        self.operator_type = None
        return None
