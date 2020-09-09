"""
A simple example inspection
"""
from typing import Union, Iterable

from mlinspect.inspections.inspection_input import OperatorContext, InspectionInputDataSource, \
    InspectionInputUnaryOperator
from mlinspect.inspections.inspection import Inspection
from mlinspect.instrumentation.dag_node import OperatorType


def get_current_annotation(row):
    """
    Get the current row annotation value
    """
    if isinstance(row, InspectionInputUnaryOperator):
        annotation = row.annotation.get_value_by_column_index(0)
    else:
        assert not isinstance(row, InspectionInputDataSource)
        annotation = row.annotation[0].get_value_by_column_index(0)
    return annotation


class HistogramInspection(Inspection):
    """
    A simple example inspection
    """

    def __init__(self):
        self._histogram_op_output = None
        self._operator_type = None

    def visit_operator(self, operator_context: OperatorContext,
                       row_iterator: Union[Iterable[InspectionInputDataSource], Iterable[InspectionInputUnaryOperator]])\
            -> Iterable[any]:
        """
        Visit an operator
        """
        # pylint: disable=too-many-branches, too-many-statements
        current_count = - 1

        age_group_map = {}
        race_count_map = {}
        histogram_map = {}

        self._operator_type = operator_context.operator

        if self._operator_type in {OperatorType.DATA_SOURCE, OperatorType.GROUP_BY_AGG}:
            for row in row_iterator:
                current_count += 1
                if "age_group" in row.output.fields:
                    age_group_index = row.output.fields.index("age_group")
                    age_group = row.output.values[age_group_index]

                    group_count = age_group_map.get(age_group, 0)
                    group_count += 1
                    age_group_map[age_group] = group_count

                if "race" in row.output.fields:
                    race_index = row.output.fields.index("race")
                    race = row.output.values[race_index]

                    group_count = race_count_map.get(race, 0)
                    group_count += 1
                    race_count_map[race] = group_count

                yield None
        elif self._operator_type in {OperatorType.PROJECTION, OperatorType.TRANSFORMER}:
            for row in row_iterator:
                current_count += 1
                if "age_group" in row.input.fields and "age_group" not in row.output.fields:
                    age_group_index = row.input.fields.index("age_group")
                    age_group = row.input.values[age_group_index]
                else:
                    age_group = get_current_annotation(row)[0]
                if "race" in row.input.fields and "race" not in row.output.fields:
                    if operator_context.function_info != ('sklearn.impute._base', 'fit_transform'):
                        race_index = row.input.fields.index("race")
                        race = row.input.values[race_index]
                    else:
                        race = row.output.values[0]
                else:
                    race = get_current_annotation(row)[1]

                group_count = age_group_map.get(age_group, 0)
                group_count += 1
                age_group_map[age_group] = group_count

                group_count = race_count_map.get(race, 0)
                group_count += 1
                race_count_map[race] = group_count

                annotation = (age_group, race)
                group_count = histogram_map.get(annotation, 0)
                group_count += 1
                histogram_map[annotation] = group_count
                yield annotation
        elif self._operator_type is not OperatorType.ESTIMATOR:
            for row in row_iterator:

                current_count += 1
                if "age_group" in row.output.fields:
                    age_group_index = row.output.fields.index("age_group")
                    age_group = row.output.values[age_group_index]
                else:
                    age_group = get_current_annotation(row)[0]
                if "race" in row.output.fields:
                    race_index = row.output.fields.index("race")
                    race = row.output.values[race_index]
                else:
                    race = get_current_annotation(row)[1]

                group_count = age_group_map.get(age_group, 0)
                group_count += 1
                age_group_map[age_group] = group_count

                group_count = race_count_map.get(race, 0)
                group_count += 1
                race_count_map[race] = group_count

                annotation = (age_group, race)
                group_count = histogram_map.get(annotation, 0)
                group_count += 1
                histogram_map[annotation] = group_count

                yield annotation
        else:
            for _ in row_iterator:
                yield None

        self._histogram_op_output = {"age_group_counts": age_group_map, "race_counts": race_count_map,
                                     "age_groups_race_counts": histogram_map}

    def get_operator_annotation_after_visit(self) -> any:
        assert self._operator_type
        if self._operator_type is not OperatorType.ESTIMATOR:
            result = self._histogram_op_output
            self._histogram_op_output = None
            self._operator_type = None
            return result
        self._operator_type = None
        return None
