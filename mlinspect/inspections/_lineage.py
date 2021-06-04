"""
A simple inspection for testing annotation propagation
"""
import dataclasses
from typing import Iterable

from pandas import DataFrame, Series

from mlinspect.inspections._inspection import Inspection
from mlinspect.inspections._inspection_input import InspectionInputUnaryOperator, \
    InspectionInputSinkOperator, InspectionInputDataSource, InspectionInputNAryOperator, OperatorType


@dataclasses.dataclass(frozen=True)
class LineageId:
    """
    A lineage id class
    """
    operator_id: int
    row_id: int


class RowLineage(Inspection):
    """
    A simple inspection for testing annotation propagation
    """
    # TODO: Add an option to pass a list of lineage ids to this inspection. Then it materializes all related tuples.
    #  To do this efficiently, we do not want to do expensive membership tests. We can collect all base LineageIds
    #  in a set and then it is enough to check for set memberships in InspectionInputDataSource inspection inputs.
    #  This set membership can be used as a 'materialize' flag we use as annotation. Then we simply need to check this
    #  flag to check whether to materialize rows.

    def __init__(self, row_count: int):
        self.row_count = row_count
        self._inspection_id = self.row_count

        self._operator_count = 0
        self._op_output = None
        self._op_lineage = None
        self._output_columns = None
        self._is_sink = False

    def visit_operator(self, inspection_input) -> Iterable[any]:
        """Visit an operator, generate row index number annotations and check whether they get propagated correctly"""
        # pylint: disable=too-many-branches
        operator_output = []
        operator_lineage = []
        current_count = -1

        if not isinstance(inspection_input, InspectionInputSinkOperator):
            self._output_columns = inspection_input.output_columns.fields

        if isinstance(inspection_input, InspectionInputDataSource):
            for row in inspection_input.row_iterator:
                current_count += 1
                annotation = {LineageId(self._operator_count, current_count)}
                if current_count < self.row_count:
                    operator_output.append(row.output)
                    operator_lineage.append(annotation)
                yield annotation

        elif isinstance(inspection_input, InspectionInputNAryOperator):
            if inspection_input.operator_context.operator == OperatorType.JOIN:
                for row in inspection_input.row_iterator:
                    current_count += 1

                    annotation = set.union(*row.annotation)
                    if current_count < self.row_count:
                        operator_output.append(row.output)
                        operator_lineage.append(annotation)
                    yield annotation
            elif inspection_input.operator_context.operator == OperatorType.CONCATENATION:
                for row in inspection_input.row_iterator:
                    current_count += 1

                    annotation = set.union(*row.annotation)
                    if current_count < self.row_count:
                        operator_output.append(row.output)
                        operator_lineage.append(annotation)
                    yield annotation
            else:
                assert False
        elif isinstance(inspection_input, InspectionInputUnaryOperator):
            for row in inspection_input.row_iterator:
                current_count += 1
                annotation = row.annotation

                if current_count < self.row_count:
                    operator_output.append(row.output)
                    operator_lineage.append(annotation)
                yield annotation
        elif isinstance(inspection_input, InspectionInputSinkOperator):
            self._is_sink = True
            for row in inspection_input.row_iterator:
                current_count += 1
                annotation = set.union(*row.annotation)
                if current_count < self.row_count:
                    operator_lineage.append(annotation)
                yield annotation
        else:
            assert False
        self._operator_count += 1
        self._op_output = operator_output
        self._op_lineage = operator_lineage

    def get_operator_annotation_after_visit(self) -> any:
        assert self._op_lineage  # May only be called after the operator visit is finished
        if not self._is_sink:
            result = DataFrame(self._op_output, columns=self._output_columns)
            result["mlinspect_lineage"] = self._op_lineage
        else:
            lineage_series = Series(self._op_lineage)
            result = DataFrame(lineage_series, columns=["mlinspect_lineage"])
        self._op_output = None
        self._op_lineage = None
        self._output_columns = None
        self._is_sink = False
        return result

    @property
    def inspection_id(self):
        return self._inspection_id
