"""
A simple example analyzer
"""
from typing import Union, Iterable

from mlinspect.instrumentation.analyzers.analyzer_input import OperatorContext, AnalyzerInputDataSource, \
    AnalyzerInputUnaryOperator
from mlinspect.instrumentation.analyzers.analyzer import Analyzer
from mlinspect.instrumentation.dag_node import OperatorType


def get_current_annotation(row):
    """
    Get the current row annotation value
    """
    if isinstance(row, AnalyzerInputUnaryOperator):
        annotation = row.annotation.get_value_by_column_index(0)
    else:
        assert not isinstance(row, AnalyzerInputDataSource)
        annotation = row.annotation[0].get_value_by_column_index(0)
    return annotation


class FairnessDemoAnalyzer(Analyzer):
    """
    A simple example analyzer
    """

    def __init__(self, row_count: int):
        self.row_count = row_count
        self._analyzer_id = self.row_count
        self._operator_output = None
        self._operator_type = None

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
        self._operator_type = operator_context.operator

        if self._operator_type in {OperatorType.DATA_SOURCE, OperatorType.GROUP_BY_AGG}:
            for _ in row_iterator:
                current_count += 1
                self.update_operator_output(None, current_count, operator_output)
                yield None
        elif self._operator_type is OperatorType.PROJECTION:
            for row in row_iterator:
                current_count += 1
                if "age_group" in row.input.fields and "age_group" not in row.output.fields:
                    age_group_index = row.input.fields.index("age_group")
                    annotation = row.input.values[age_group_index]
                else:
                    annotation = get_current_annotation(row)
                self.update_operator_output(annotation, current_count, operator_output)
                yield annotation
        elif self._operator_type is not OperatorType.ESTIMATOR:
            for row in row_iterator:
                current_count += 1
                annotation = get_current_annotation(row)
                self.update_operator_output(annotation, current_count, operator_output)
                yield annotation
        else:
            for _ in row_iterator:
                operator_output.append(None)
                yield None

        self._operator_output = operator_output

    def update_operator_output(self, annotation, current_count, operator_output):
        """
        Append the first annotations to the operator output to showcase that it works
        """
        if current_count < self.row_count:
            operator_output.append(annotation)

    def get_operator_annotation_after_visit(self) -> any:
        assert self._operator_type
        if self._operator_type is not OperatorType.ESTIMATOR:
            assert self._operator_output  # May only be called after the operator visit is finished
            result = self._operator_output
            self._operator_output = None
            self._operator_type = None
            return result
        self._operator_type = None
        return None
