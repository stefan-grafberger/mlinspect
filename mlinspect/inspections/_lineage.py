"""
A simple inspection for lineage tracking
"""
import dataclasses
from typing import Iterable, List

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
    A simple inspection for row-level lineage tracking
    """
    # TODO: Add an option to pass a list of lineage ids to this inspection. Then it materializes all related tuples.
    #  To do this efficiently, we do not want to do expensive membership tests. We can collect all base LineageIds
    #  in a set and then it is enough to check for set memberships in InspectionInputDataSource inspection inputs.
    #  This set membership can be used as a 'materialize' flag we use as annotation. Then we simply need to check this
    #  flag to check whether to materialize rows.
    # pylint: disable=too-many-instance-attributes

    ALL_ROWS = -1

    def __init__(self, row_count: int, operator_type_restriction: List[OperatorType] = None):
        self.row_count = row_count
        if operator_type_restriction is not None:
            self.operator_type_restriction = set(operator_type_restriction)
            self._inspection_id = (self.row_count, *self.operator_type_restriction)
        else:
            self.operator_type_restriction = None
            self._inspection_id = self.row_count
        self._operator_count = -1
        self._op_output = None
        self._op_lineage = None
        self._output_columns = None
        self._is_sink = False
        self._materialize_for_this_operator = None

    def visit_operator(self, inspection_input) -> Iterable[any]:
        """Visit an operator, generate row index number annotations and check whether they get propagated correctly"""
        # pylint: disable=too-many-branches, too-many-statements
        self._operator_count += 1
        self._op_output = []
        self._op_lineage = []
        current_count = -1
        self._materialize_for_this_operator = (self.operator_type_restriction is None) or \
                                              (inspection_input.operator_context.operator
                                               in self.operator_type_restriction)

        if not isinstance(inspection_input, InspectionInputSinkOperator):
            self._output_columns = inspection_input.output_columns.fields
        else:
            self._is_sink = True

        if isinstance(inspection_input, InspectionInputDataSource):
            if self._materialize_for_this_operator and self.row_count == RowLineage.ALL_ROWS:
                for row in inspection_input.row_iterator:
                    current_count += 1
                    annotation = {LineageId(self._operator_count, current_count)}
                    self._op_output.append(row.output)
                    self._op_lineage.append(annotation)
                    yield annotation
            elif self._materialize_for_this_operator:
                for row in inspection_input.row_iterator:
                    current_count += 1
                    annotation = {LineageId(self._operator_count, current_count)}
                    if current_count < self.row_count:
                        self._op_output.append(row.output)
                        self._op_lineage.append(annotation)
                    yield annotation
            else:
                for _ in inspection_input.row_iterator:
                    current_count += 1
                    annotation = {LineageId(self._operator_count, current_count)}
                    yield annotation
        elif isinstance(inspection_input, InspectionInputNAryOperator):
            if self._materialize_for_this_operator and self.row_count == RowLineage.ALL_ROWS:
                for row in inspection_input.row_iterator:
                    current_count += 1

                    annotation = set.union(*row.annotation)
                    self._op_output.append(row.output)
                    self._op_lineage.append(annotation)
                    yield annotation
            elif self._materialize_for_this_operator:
                for row in inspection_input.row_iterator:
                    current_count += 1

                    annotation = set.union(*row.annotation)
                    if current_count < self.row_count:
                        self._op_output.append(row.output)
                        self._op_lineage.append(annotation)
                    yield annotation
            else:
                for row in inspection_input.row_iterator:
                    current_count += 1
                    annotation = set.union(*row.annotation)
                    yield annotation
        elif isinstance(inspection_input, InspectionInputSinkOperator):
            if self._materialize_for_this_operator and self.row_count == RowLineage.ALL_ROWS:
                for row in inspection_input.row_iterator:
                    current_count += 1

                    annotation = set.union(*row.annotation)
                    self._op_lineage.append(annotation)
                    yield annotation
            elif self._materialize_for_this_operator:
                for row in inspection_input.row_iterator:
                    current_count += 1

                    annotation = set.union(*row.annotation)
                    if current_count < self.row_count:
                        self._op_lineage.append(annotation)
                    yield annotation
            else:
                for row in inspection_input.row_iterator:
                    current_count += 1
                    annotation = set.union(*row.annotation)
                    yield annotation
        elif isinstance(inspection_input, InspectionInputUnaryOperator):
            if self._materialize_for_this_operator and self.row_count == RowLineage.ALL_ROWS:
                for row in inspection_input.row_iterator:
                    current_count += 1
                    annotation = row.annotation
                    self._op_output.append(row.output)
                    self._op_lineage.append(annotation)
                    yield annotation
            elif self._materialize_for_this_operator:
                for row in inspection_input.row_iterator:
                    current_count += 1
                    annotation = row.annotation

                    if current_count < self.row_count:
                        self._op_output.append(row.output)
                        self._op_lineage.append(annotation)
                    yield annotation
            else:
                for row in inspection_input.row_iterator:
                    current_count += 1
                    annotation = row.annotation
                    yield annotation
        else:
            assert False

    def get_operator_annotation_after_visit(self) -> any:
        if not self._materialize_for_this_operator:
            result = None
        elif not self._is_sink:
            assert self._op_lineage
            result = DataFrame(self._op_output, columns=self._output_columns)
            result["mlinspect_lineage"] = self._op_lineage
        else:
            assert self._op_lineage
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
