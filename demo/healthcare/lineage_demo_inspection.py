"""
A simple inspection for testing annotation propagation
"""
import dataclasses
from typing import Iterable, List

from mlinspect.inspections.inspection import Inspection
from mlinspect.inspections.inspection_input import InspectionInputUnaryOperator, \
    InspectionInputSinkOperator, InspectionInputDataSource, InspectionInputNAryOperator
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

    def visit_operator(self, inspection_input) -> Iterable[any]:
        """Visit an operator, generate row index number annotations and check whether they get propagated correctly"""
        # pylint: disable=too-many-branches
        operator_output = []
        current_count = -1

        if isinstance(inspection_input, InspectionInputDataSource):
            for row in inspection_input.row_iterator:
                current_count += 1
                annotation = LineageId(self._operator_count, current_count)
                if current_count < self.row_count:
                    operator_output.append((annotation, row.output))
                yield annotation

        elif isinstance(inspection_input, InspectionInputNAryOperator):
            if inspection_input.operator_context.operator == OperatorType.JOIN:
                for row in inspection_input.row_iterator:
                    current_count += 1

                    annotation = JoinLineageId(list(*row.annotation))
                    if current_count < self.row_count:
                        operator_output.append((annotation, row.output))
                    yield annotation
            elif inspection_input.operator_context.operator == OperatorType.CONCATENATION:
                for row in inspection_input.row_iterator:
                    current_count += 1

                    annotation = ConcatLineageId(list(*row.annotation))
                    if current_count < self.row_count:
                        operator_output.append((annotation, row.output))
                    yield annotation
            else:
                assert False
        elif isinstance(inspection_input, InspectionInputUnaryOperator):
            for row in inspection_input.row_iterator:
                current_count += 1
                annotation = row.annotation

                if current_count < self.row_count:
                    operator_output.append((annotation, row.output))
                yield annotation
        elif isinstance(inspection_input, InspectionInputSinkOperator):
            for row in inspection_input.row_iterator:
                current_count += 1
                annotation = row.annotation
                operator_output.append((annotation, None))
                yield annotation
        else:
            assert False
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
