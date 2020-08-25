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


class MissingEmbeddingInspection(Analyzer):
    """
    A simple example analyzer
    """

    def __init__(self):
        self._is_embedding_operator = False
        self._missing_embedding_count = 0

    @property
    def analyzer_id(self):
        return None

    def visit_operator(self, operator_context: OperatorContext,
                       row_iterator: Union[Iterable[AnalyzerInputDataSource], Iterable[AnalyzerInputUnaryOperator]])\
            -> Iterable[any]:
        """
        Visit an operator
        """
        # pylint: disable=too-many-branches, too-many-statements
        if operator_context.operator is OperatorType.TRANSFORMER:
            for row in row_iterator:

                # Count missing embeddings
                if operator_context.function_info == ('demo.healthcare.demo_utils', 'fit_transform'):
                    self._is_embedding_operator = True
                    embedding_array = row.output.values[0]
                    is_zero_vector = not embedding_array.any()
                    if is_zero_vector:
                        self._missing_embedding_count += 1
                yield None
        else:
            for _ in row_iterator:
                yield None

    def get_operator_annotation_after_visit(self) -> any:
        if self._is_embedding_operator:
            assert self._missing_embedding_count is not None  # May only be called after the operator visit is finished
            result = self._missing_embedding_count
            self._missing_embedding_count = None
            self._is_embedding_operator = False
            return result
        return None
