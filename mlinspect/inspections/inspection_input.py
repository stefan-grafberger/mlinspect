"""
Data classes used as input for the inspections
"""
import dataclasses
from typing import Tuple, List

import numpy

from ..instrumentation.dag_node import OperatorType


@dataclasses.dataclass(frozen=True)
class InspectionInputRow:
    """
    A class we use to efficiently pass pandas/sklearn rows
    """
    values: list
    fields: list

    def get_index_of_column(self, column_name):
        """
        Get the values index for some column
        """
        if column_name in self.fields:
            return self.fields.index(column_name)
        return None

    def get_value_by_column_index(self, index):
        """
        Get the value at some index
        """
        return self.values[index]

    def __eq__(self, other):
        return (isinstance(other, InspectionInputRow) and
                numpy.array_equal(self.values, other.values) and
                self.fields == other.fields)


@dataclasses.dataclass(frozen=True)
class InspectionInputDataSource:
    """
    Wrapper class for the only operator without a parent: a Data Source
    """
    output: InspectionInputRow


@dataclasses.dataclass(frozen=True)
class InspectionInputUnaryOperator:
    """
    Wrapper class for the operators with one parent like Selections and Projections
    """
    input: InspectionInputRow
    annotation: InspectionInputRow
    output: InspectionInputRow


@dataclasses.dataclass(frozen=True)
class InspectionInputNAryOperator:
    """
    Wrapper class for the operators with multiple parents like Concatenations
    """
    input: List[InspectionInputRow]
    annotation: List[InspectionInputRow]
    output: InspectionInputRow


@dataclasses.dataclass(frozen=True)
class InspectionInputSinkOperator:
    """
    Wrapper class for operators like Estimators that only get fitted
    """
    input: List[InspectionInputRow]
    annotation: List[InspectionInputRow]


@dataclasses.dataclass(frozen=True)
class OperatorContext:
    """
    Additional context for the inspection. Contains, most importantly, the operator type.
    """
    operator: OperatorType
    function_info: Tuple[str, str]
