"""
A simple inspection for testing annotation propagation
"""
import dataclasses
from typing import Iterable, List

from mlinspect.inspections.inspection import Inspection
from mlinspect.inspections.inspection_input import OperatorContext, InspectionInputUnaryOperator, \
    InspectionInputSinkOperator
from mlinspect.instrumentation.dag_node import OperatorType


@dataclasses.dataclass(frozen=True)
class LineageId:
    """
    A lineage id class
    """
    operator_id: int
    row_id: int


@dataclasses.dataclass(frozen=True)
class JoinLineageId:
    """
    A lineage id class
    """
    lineage_ids: List


@dataclasses.dataclass(frozen=True)
class ConcatLineageId:
    """
    A lineage id class
    """
    lineage_ids: List


class LineageDemoInspection(Inspection):
    """
    A simple inspection for testing annotation propagation
    """

    def __init__(self, row_count: int):
        self.row_count = row_count
        self._inspection_id = self.row_count

        self._operator_count = 0
        self._op_output = None

    def visit_operator(self, operator_context: OperatorContext, row_iterator) -> Iterable[any]:
        """Visit an operator, generate row index number annotations and check whether they get propagated correctly"""
        # pylint: disable=too-many-branches
        operator_output = []
        current_count = -1

        if operator_context.operator in {OperatorType.DATA_SOURCE, OperatorType.GROUP_BY_AGG}:
            for row in row_iterator:
                current_count += 1
                annotation = LineageId(self._operator_count, current_count)
                if current_count < self.row_count:
                    operator_output.append((annotation, row.output))
                yield annotation

        elif operator_context.operator in {OperatorType.JOIN}:
            for row in row_iterator:
                current_count += 1

                parent_annotations = [annotation.values[0] for annotation in row.annotation]
                annotation = JoinLineageId(parent_annotations)
                if current_count < self.row_count:
                    operator_output.append((annotation, row.output))
                yield annotation
        elif operator_context.operator in {OperatorType.CONCATENATION}:
            for row in row_iterator:
                current_count += 1

                parent_annotations = [annotation.values[0] for annotation in row.annotation]
                annotation = ConcatLineageId(parent_annotations)
                if current_count < self.row_count:
                    operator_output.append((annotation, row.output))
                yield annotation
        else:
            for row in row_iterator:
                current_count += 1

                if isinstance(row, InspectionInputUnaryOperator):
                    annotation = row.annotation.get_value_by_column_index(0)
                elif isinstance(row, InspectionInputSinkOperator):
                    annotation = row.annotation[0].get_value_by_column_index(0)
                else:
                    assert False

                if current_count < self.row_count and not isinstance(row, InspectionInputSinkOperator):
                    operator_output.append((annotation, row.output))
                elif current_count < self.row_count:
                    operator_output.append((annotation, None))
                yield annotation
        self._operator_count += 1
        self._op_output = operator_output

    def get_operator_annotation_after_visit(self) -> any:
        assert self._op_output  # May only be called after the operator visit is finished
        result = self._op_output
        self._op_output = None
        return result

    @property
    def inspection_id(self):
        return self._inspection_id
