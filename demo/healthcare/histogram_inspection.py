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
        # pylint: disable=too-many-branches, too-many-statements
        current_count = - 1

        age_group_map = {}
        race_count_map = {}

        self._operator_type = inspection_input.operator_context.operator

        if isinstance(inspection_input, InspectionInputUnaryOperator):
            age_group = [None, None]
            if inspection_input.operator_context.function_info == ('sklearn.impute._base', 'fit_transform') and \
                    "age_group" in inspection_input.input_columns.fields:
                age_group_index = inspection_input.input_columns.get_index_of_column("age_group")
                age_group_output = 0
                age_group_annotation = 1
            elif "age_group" in inspection_input.output_columns.fields:
                age_group_index = inspection_input.output_columns.get_index_of_column("age_group")
                age_group_output = 0
                age_group_annotation = 1
            else:
                age_group_index = 0
                age_group_output = 1
                age_group_annotation = 0
            race = [None, None]
            if inspection_input.operator_context.function_info == ('sklearn.impute._base', 'fit_transform') and \
                    "race" in inspection_input.input_columns.fields:
                race_index = inspection_input.input_columns.get_index_of_column("race")
                race_output = 0
                race_annotation = 1
            elif "race" in inspection_input.output_columns.fields:
                race_index = inspection_input.output_columns.get_index_of_column("race")
                race_output = 0
                race_annotation = 1
            else:
                race_index = 0
                race_output = 1
                race_annotation = 0
            for row in inspection_input.row_iterator:
                current_count += 1
                age_group[age_group_output] = row.input[age_group_index]
                age_group[age_group_annotation] = row.annotation[0]
                race[race_output] = row.input[race_index]
                race[race_annotation] = row.annotation[1]

                annotation = (age_group[0], race[0])
                self.update_histogram_maps(age_group[0], age_group_map, race[0], race_count_map)
                yield annotation
        elif isinstance(inspection_input, InspectionInputDataSource):
            if "age_group" in inspection_input.output_columns.fields:
                age_group_index = inspection_input.output_columns.get_index_of_column("age_group")
                age_group_present = True
            else:
                age_group_present = False
            if "race" in inspection_input.output_columns.fields:
                race_index = inspection_input.output_columns.get_index_of_column("age_group")
                race_present = True
            else:
                race_present = False
            for row in inspection_input.row_iterator:
                current_count += 1
                if age_group_present:
                    age_group = row.output[age_group_index]
                    group_count = age_group_map.get(age_group, 0)
                    group_count += 1
                    age_group_map[age_group] = group_count
                if race_present:
                    race_group = row.output[race_index]
                    group_count = race_count_map.get(race_group, 0)
                    group_count += 1
                    race_count_map[race_group] = group_count
                yield None
        elif isinstance(inspection_input, InspectionInputNAryOperator):
            if "age_group" in inspection_input.output_columns.fields:
                age_group_index = inspection_input.output_columns.get_index_of_column("age_group")
                age_group_present = True
            else:
                age_group_present = False
            if "race" in inspection_input.output_columns.fields:
                race_index = inspection_input.output_columns.get_index_of_column("age_group")
                race_present = True
            else:
                race_present = False
            for row in inspection_input.row_iterator:
                current_count += 1
                if age_group_present:
                    age_group = row.output[age_group_index]
                else:
                    age_group = row.annotation[0][0]
                if race_present:
                    race = row.output[race_index]
                else:
                    race = row.annotation[0][1]
                yield age_group, race
        else:
            for _ in inspection_input.row_iterator:
                yield None

        self._histogram_op_output = {"age_group_counts": age_group_map, "race_counts": race_count_map}

    @staticmethod
    def update_histogram_maps(age_group, age_group_map, race, race_count_map):
        """
        Update the histogram maps.
        """
        group_count = age_group_map.get(age_group, 0)
        group_count += 1
        age_group_map[age_group] = group_count
        group_count = race_count_map.get(race, 0)
        group_count += 1
        race_count_map[race] = group_count

    def get_operator_annotation_after_visit(self) -> any:
        assert self._operator_type
        if self._operator_type is not OperatorType.ESTIMATOR:
            result = self._histogram_op_output
            self._histogram_op_output = None
            self._operator_type = None
            return result
        self._operator_type = None
        return None
