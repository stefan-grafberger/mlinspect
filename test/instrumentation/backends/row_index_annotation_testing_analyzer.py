"""
A simple analyzer for testing annotation propagation
"""
from typing import Iterable

from mlinspect.instrumentation.analyzers.analyzer import Analyzer
from mlinspect.instrumentation.analyzers.analyzer_input import OperatorContext, AnalyzerInputUnaryOperator, \
    AnalyzerInputNAryOperator, AnalyzerInputSinkOperator


class RowIndexAnnotationTestingAnalyzer(Analyzer):
    """
    A simple analyzer for testing annotation propagation
    """

    def __init__(self, row_count: int):
        self.row_count = row_count
        self._analyzer_id = self.row_count

        self._operator_count = 0
        self._operator_output = None
        self._operator_type = None

    def visit_operator(self, operator_context: OperatorContext, row_iterator) -> Iterable[any]:
        """Visit an operator, generate row index number annotations and check whether they get propagated correctly"""
        # pylint: disable=too-many-branches
        self._operator_type = operator_context.operator
        operator_output = []
        current_count = -1

        if self._operator_count == 0:
            for row in row_iterator:
                current_count += 1
                annotation = (self._operator_count, current_count)
                if current_count < self.row_count:
                    operator_output.append((annotation, row.output))
                yield annotation
        else:
            for row in row_iterator:
                current_count += 1
                assert isinstance(row, (AnalyzerInputUnaryOperator, AnalyzerInputNAryOperator,
                                        AnalyzerInputSinkOperator))
                if isinstance(row, AnalyzerInputUnaryOperator):
                    annotation = (self._operator_count, row.annotation.get_value_by_column_index(0)[1])
                else:
                    previous_row_index = row.annotation[0].get_value_by_column_index(0)[1]
                    annotation = (self._operator_count, previous_row_index)
                if current_count < self.row_count and not isinstance(row, AnalyzerInputSinkOperator):
                    operator_output.append((annotation, row.output))
                elif current_count < self.row_count:
                    operator_output.append((annotation, None))
                yield annotation
        self._operator_count += 1
        self._operator_output = operator_output

    def get_operator_annotation_after_visit(self) -> any:
        assert self._operator_type and self._operator_output  # May only be called after the operator visit is finished
        self._operator_type = None
        self._operator_output = None
        result = self._operator_output
        return result

    @property
    def analyzer_id(self):
        return self._analyzer_id
