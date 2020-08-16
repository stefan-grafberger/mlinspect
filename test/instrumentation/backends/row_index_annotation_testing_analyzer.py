"""
A simple analyzer for testing annotation propagation
"""
from typing import Iterable, Union

from mlinspect.instrumentation.analyzers.analyzer_input import OperatorContext, AnalyzerInputDataSource, \
    AnalyzerInputUnaryOperator, AnalyzerInputNAryOperator, AnalyzerInputSinkOperator
from mlinspect.instrumentation.analyzers.analyzer import Analyzer


class RowIndexAnnotationTestingAnalyzer(Analyzer):
    """
    A simple analyzer for testing annotation propagation
    """

    def __init__(self, row_count: int):
        self.row_count = row_count
        self._operator_count = 0
        self._analyzer_id = self.row_count
        self._operator_output = None
        self._operator_type = None
        self._rows_to_random_numbers_operator_0 = {}

    def visit_operator(self, operator_context: OperatorContext,
                       row_iterator: Union[Iterable[AnalyzerInputDataSource], Iterable[AnalyzerInputUnaryOperator]])\
            -> Iterable[any]:
        """Visit an operator, generate random number annotations and check whether they get propagated correctly"""
        # pylint: disable=too-many-branches
        operator_output = []
        current_count = - 1
        self._operator_type = operator_context.operator

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
        assert self._operator_type
        assert self._operator_output  # May only be called after the operator visit is finished
        result = self._operator_output
        self._operator_output = None
        self._operator_type = None
        return result

    @property
    def analyzer_id(self):
        return self._analyzer_id
