"""
A simple analyzer for testing annotation propagation
"""
import random
from typing import Iterable, Union

from mlinspect.instrumentation.analyzers.analyzer_input import OperatorContext, AnalyzerInputDataSource, \
    AnalyzerInputUnaryOperator
from mlinspect.instrumentation.analyzers.analyzer import Analyzer


class AnnotationTestingAnalyzer(Analyzer):
    """
    A simple analyzer for testing annotation propagation
    """

    def __init__(self, row_count: int):
        self.operator_count = 0
        self.row_count = row_count
        self._analyzer_id = self.row_count
        self._operator_output = None
        self.rows_to_random_numbers_operator_0 = {}

    def visit_operator(self, operator_context: OperatorContext,
                       row_iterator: Union[Iterable[AnalyzerInputDataSource], Iterable[AnalyzerInputUnaryOperator]])\
            -> Iterable[any]:
        """Visit an operator, generate random number annotations and check whether they get propagated correctly"""
        operator_output = []
        current_count = - 1

        if self.operator_count == 0:
            for row in row_iterator:
                current_count += 1
                if current_count < self.row_count:
                    random_number = random.randint(0, 10000)
                    output_tuple = tuple(row.output.values)
                    self.rows_to_random_numbers_operator_0[output_tuple] = random_number
                    operator_output.append(row.output)
                    yield random_number
                else:
                    yield None
        elif self.operator_count == 1:
            filtered_rows = 0
            for row in row_iterator:
                current_count += 1
                assert isinstance(row, AnalyzerInputUnaryOperator)  # This analyzer is really only for testing
                annotation = row.annotation.get_value_by_column_index(0)
                if current_count < self.row_count:
                    output_tuple = tuple(row.output.values)
                    if output_tuple in self.rows_to_random_numbers_operator_0:
                        random_number = self.rows_to_random_numbers_operator_0[output_tuple]
                        assert annotation == random_number  # Test whether the annotation got propagated correctly
                    else:
                        filtered_rows += 1
                    assert filtered_rows != self.row_count  # If all rows got filtered, this test is useless
                    operator_output.append(row.output)
                yield annotation
        else:
            yield None
        self.operator_count += 1
        self._operator_output = operator_output

    def get_operator_annotation_after_visit(self) -> any:
        assert self._operator_output or self.operator_count > 1
        result = self._operator_output
        self._operator_output = None
        return result

    @property
    def analyzer_id(self):
        return self._analyzer_id
