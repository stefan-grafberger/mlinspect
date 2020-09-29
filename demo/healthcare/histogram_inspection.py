"""
A simple example inspection
"""
from typing import Iterable

from mlinspect.inspections.inspection import Inspection
from mlinspect.inspections.inspection_input import InspectionInputDataSource, \
    InspectionInputUnaryOperator, InspectionInputNAryOperator
from mlinspect.instrumentation.dag_node import OperatorType


class HistogramInspection(Inspection):
    """
    A simple example inspection
    """

    def __init__(self):
        self._histogram_op_output = None
        self._operator_type = None

    def visit_operator(self, inspection_input) -> Iterable[any]:
        """
        Visit an operator
        """
        # pylint: disable=too-many-branches, too-many-statements, too-many-locals
        current_count = - 1

        age_group_map = {}
        race_count_map = {}

        self._operator_type = inspection_input.operator_context.operator

        if isinstance(inspection_input, InspectionInputUnaryOperator):
            age_group_index = inspection_input.input_columns.get_index_of_column("age_group")
            age_group_present = "age_group" in inspection_input.input_columns.fields
            race_index = inspection_input.input_columns.get_index_of_column("race")
            race_present = "race" in inspection_input.input_columns.fields
            if inspection_input.operator_context.function_info == ('sklearn.impute._base', 'fit_transform'):
                for row in inspection_input.row_iterator:
                    current_count += 1
                    if age_group_present:
                        age_group = row.output[age_group_index][0]
                    else:
                        age_group = row.annotation[0]
                    self.update_histogram_map(age_group, age_group_map)
                    if race_present:
                        race = row.output[race_index][0]
                    else:
                        race = row.annotation[1]
                    self.update_histogram_map(race, race_count_map)
                    yield age_group, race
            else:
                for row in inspection_input.row_iterator:
                    current_count += 1
                    if age_group_present:
                        age_group = row.input[age_group_index]
                    else:
                        age_group = row.annotation[0]
                    self.update_histogram_map(age_group, age_group_map)
                    if race_present:
                        race = row.input[race_index]
                    else:
                        race = row.annotation[1]
                    self.update_histogram_map(race, race_count_map)
                    yield age_group, race
        elif isinstance(inspection_input, InspectionInputDataSource):
            age_group_index = inspection_input.output_columns.get_index_of_column("age_group")
            age_group_present = "age_group" in inspection_input.output_columns.fields
            race_index = inspection_input.output_columns.get_index_of_column("race")
            race_present = "race" in inspection_input.output_columns.fields
            for row in inspection_input.row_iterator:
                current_count += 1
                if age_group_present:
                    age_group = row.output[age_group_index]
                    self.update_histogram_map(age_group, age_group_map)
                if race_present:
                    race = row.output[race_index]
                    self.update_histogram_map(race, race_count_map)
                yield None
        elif isinstance(inspection_input, InspectionInputNAryOperator):
            age_group_index = inspection_input.output_columns.get_index_of_column("age_group")
            age_group_present = "age_group" in inspection_input.output_columns.fields
            race_index = inspection_input.output_columns.get_index_of_column("race")
            race_present = "race" in inspection_input.output_columns.fields
            for row in inspection_input.row_iterator:
                current_count += 1
                if age_group_present:
                    age_group = row.output[age_group_index]
                else:
                    age_group = row.annotation[0][0]
                self.update_histogram_map(age_group, age_group_map)
                if race_present:
                    race = row.output[race_index]
                else:
                    race = row.annotation[0][1]
                self.update_histogram_map(race, race_count_map)
                yield age_group, race
        else:
            for _ in inspection_input.row_iterator:
                yield None

        self._histogram_op_output = {"age_group_counts": age_group_map, "race_counts": race_count_map}

    @staticmethod
    def update_histogram_map(group, histogram_map):
        """
        Update the histogram maps.
        """
        group_count = histogram_map.get(group, 0)
        group_count += 1
        histogram_map[group] = group_count

    def get_operator_annotation_after_visit(self) -> any:
        assert self._operator_type
        if self._operator_type is not OperatorType.ESTIMATOR:
            result = self._histogram_op_output
            self._histogram_op_output = None
            self._operator_type = None
            return result
        self._operator_type = None
        return None
