"""
A simple example inspection
"""
from typing import Union, Iterable
from mlinspect.inspections.inspection_input import OperatorContext, InspectionInputDataSource, \
    InspectionInputUnaryOperator
from mlinspect.instrumentation.dag_node import OperatorType
from mlinspect.inspections.inspection import Inspection


class MissingEmbeddingInspection(Inspection):
    """
    A simple example inspection
    """

    def __init__(self):
        self._is_embedding_operator = False
        self._missing_embedding_count = 0

    def visit_operator(self, operator_context: OperatorContext,
                       row_iterator: Union[Iterable[InspectionInputDataSource], Iterable[InspectionInputUnaryOperator]])\
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
