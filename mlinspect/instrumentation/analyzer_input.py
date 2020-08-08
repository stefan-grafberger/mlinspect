"""
Data classes used as input for the analyzers
"""
import dataclasses
from typing import Tuple

from mlinspect.instrumentation.dag_node import OperatorType


@dataclasses.dataclass(frozen=True)
class AnalyzerInputRow:
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


@dataclasses.dataclass(frozen=True)
class AnalyzerInputDataSource:
    """
    Wrapper class for the only operator without a parent: a Data Source
    """
    output: AnalyzerInputRow


@dataclasses.dataclass(frozen=True)
class AnalyzerInputUnaryOperator:
    """
    Wrapper class for the operators with one parent like Selections and Projections
    """
    input: AnalyzerInputRow
    annotation: AnalyzerInputRow
    output: AnalyzerInputRow


@dataclasses.dataclass(frozen=True)
class OperatorContext:
    """
    Wrapper class for the operators with one parent like Selections and Projections
    """
    operator: OperatorType
    function_info: Tuple[str, str]
