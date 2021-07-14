"""
A simple inspection to forward-propagate columns
"""
from typing import Iterable

import pandas

from mlinspect.inspections._inspection import Inspection
from mlinspect.inspections._inspection_input import InspectionInputSinkOperator, InspectionInputDataSource, \
    InspectionInputUnaryOperator, InspectionInputNAryOperator, FunctionInfo


class ColumnPropagation(Inspection):
    """
    An inspection to forward-propagate sensitive columns
    """

    def __init__(self, sensitive_columns, row_count: int):
        self.row_count = row_count
        self._operator_type = None
        self.sensitive_columns = sensitive_columns
        self._op_output = None
        self._op_annotations = None
        self._output_columns = None
        self._is_sink = False

    @property
    def inspection_id(self):
        return tuple([self.row_count, *self.sensitive_columns])

    def visit_operator(self, inspection_input) -> Iterable[any]:
        """
        Visit an operator
        """
        # pylint: disable=too-many-branches, too-many-statements, too-many-locals, too-many-nested-blocks
        operator_output = []
        operator_annotations = []
        current_count = -1

        if not isinstance(inspection_input, InspectionInputSinkOperator):
            self._output_columns = inspection_input.output_columns.fields

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
                    if current_count < self.row_count:
                        operator_output.append(row.output)
                        operator_annotations.append(column_values)
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
                    if current_count < self.row_count:
                        operator_output.append(row.output)
                        operator_annotations.append(column_values)
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
                if current_count < self.row_count:
                    operator_output.append(row.output)
                    operator_annotations.append(column_values)
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
                        if sensitive_columns_present[check_index]:
                            column_value = row.output[sensitive_columns_index[check_index]]
                        else:
                            column_value_candidates = [annotation[check_index] for annotation in row.annotation
                                                       if annotation[check_index] is not None]
                            if len(column_value_candidates) >= 1:
                                column_value = column_value_candidates[0]
                            else:
                                column_value = None
                        column_values.append(column_value)
                if current_count < self.row_count:
                    operator_output.append(row.output)
                    operator_annotations.append(column_values)
                yield column_values
        elif isinstance(inspection_input, InspectionInputSinkOperator):
            self._is_sink = True
            for row in inspection_input.row_iterator:
                current_count += 1
                column_values = []
                for check_index, _ in enumerate(self.sensitive_columns):
                    column_value_candidates = [annotation[check_index] for annotation in row.annotation
                                               if annotation[check_index] is not None]
                    if len(column_value_candidates) >= 1:
                        column_value = column_value_candidates[0]
                    else:
                        column_value = None
                    column_values.append(column_value)
                if current_count < self.row_count:
                    operator_annotations.append(column_values)
                yield column_values
        else:
            assert False

        self._op_output = operator_output
        self._op_annotations = operator_annotations

    def get_operator_annotation_after_visit(self) -> any:
        assert self._op_annotations  # May only be called after the operator visit is finished
        new_sensitive_column_names = [f"mlinspect_{column_name}" for column_name in self.sensitive_columns]
        if not self._is_sink:
            original_output_df = pandas.DataFrame(self._op_output, columns=self._output_columns)
            output_annotations = pandas.DataFrame(self._op_annotations, columns=new_sensitive_column_names)
            result = pandas.concat([original_output_df, output_annotations], axis=1)
        else:
            result = pandas.DataFrame(self._op_annotations, columns=new_sensitive_column_names)
        self._op_output = None
        self._op_annotations = None
        self._output_columns = None
        self._is_sink = False
        return result
