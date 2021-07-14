"""
An inspection to compute histograms of intersectional group memberships
"""
from typing import Iterable, List

from mlinspect.inspections._inspection import Inspection
from mlinspect.inspections._inspection_input import InspectionInputDataSource, \
    InspectionInputUnaryOperator, InspectionInputNAryOperator, OperatorType, FunctionInfo


class IntersectionalHistogramForColumns(Inspection):
    """
    An inspection to compute intersectional group memberships
    """

    def __init__(self, sensitive_columns: List[str]):
        self._histogram_op_output = None
        self._operator_type = None
        self.sensitive_columns = sensitive_columns

    @property
    def inspection_id(self):
        return tuple(self.sensitive_columns)

    def visit_operator(self, inspection_input) -> Iterable[any]:
        """
        Visit an operator
        """
        # pylint: disable=too-many-branches, too-many-statements, too-many-locals
        current_count = - 1

        histogram_map = {}

        self._operator_type = inspection_input.operator_context.operator

        if isinstance(inspection_input, InspectionInputUnaryOperator):
            sensitive_columns_present = []
            sensitive_columns_index = []
            for column in self.sensitive_columns:
                column_present = column in inspection_input.input_columns.fields
                sensitive_columns_present.append(column_present)
                column_index = inspection_input.input_columns.get_index_of_column(column)
                sensitive_columns_index.append(column_index)
            if inspection_input.operator_context.function_info == FunctionInfo('sklearn.impute._base', 'SimpleImputer'):
                for row in inspection_input.row_iterator:
                    current_count += 1
                    column_values = []
                    for check_index, _ in enumerate(self.sensitive_columns):
                        if sensitive_columns_present[check_index]:
                            column_value = row.output[0][sensitive_columns_index[check_index]]
                        else:
                            column_value = row.annotation[check_index]
                        column_values.append(column_value)
                    update_histograms(column_values, histogram_map)
                    yield column_values
            else:
                for row in inspection_input.row_iterator:
                    current_count += 1
                    column_values = []
                    for check_index, _ in enumerate(self.sensitive_columns):
                        if sensitive_columns_present[check_index]:
                            column_value = row.input[sensitive_columns_index[check_index]]
                        else:
                            column_value = row.annotation[check_index]
                        column_values.append(column_value)
                    update_histograms(column_values, histogram_map)
                    yield column_values
        elif isinstance(inspection_input, InspectionInputDataSource):
            sensitive_columns_present = []
            sensitive_columns_index = []
            for column in self.sensitive_columns:
                column_present = column in inspection_input.output_columns.fields
                sensitive_columns_present.append(column_present)
                column_index = inspection_input.output_columns.get_index_of_column(column)
                sensitive_columns_index.append(column_index)
            for row in inspection_input.row_iterator:
                current_count += 1
                column_values = []
                for check_index, _ in enumerate(self.sensitive_columns):
                    if sensitive_columns_present[check_index]:
                        column_value = row.output[sensitive_columns_index[check_index]]
                        column_values.append(column_value)
                    else:
                        column_values.append(None)
                update_histograms(column_values, histogram_map)
                yield column_values
        elif isinstance(inspection_input, InspectionInputNAryOperator):
            sensitive_columns_present = []
            sensitive_columns_index = []
            for column in self.sensitive_columns:
                column_present = column in inspection_input.output_columns.fields
                sensitive_columns_present.append(column_present)
                column_index = inspection_input.output_columns.get_index_of_column(column)
                sensitive_columns_index.append(column_index)
            for row in inspection_input.row_iterator:
                current_count += 1
                column_values = []
                for check_index, _ in enumerate(self.sensitive_columns):
                    if sensitive_columns_present[check_index]:
                        column_value = row.output[sensitive_columns_index[check_index]]
                        column_values.append(column_value)
                    else:
                        column_values.append(None)
                update_histograms(column_values, histogram_map)
                yield column_values
        else:
            for _ in inspection_input.row_iterator:
                yield None

        self._histogram_op_output = histogram_map

    def get_operator_annotation_after_visit(self) -> any:
        assert self._operator_type
        if self._operator_type is not OperatorType.ESTIMATOR:
            result = self._histogram_op_output
            self._histogram_op_output = None
            self._operator_type = None
            return result
        self._operator_type = None
        return None


def update_histograms(column_values, histogram_map):
    """Update the histograms with the intersectional information"""
    value_tuple = tuple(column_values)
    group_count = histogram_map.get(value_tuple, 0)
    group_count += 1
    histogram_map[value_tuple] = group_count
