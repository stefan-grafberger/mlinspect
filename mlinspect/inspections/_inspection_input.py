"""
Data classes used as input for the inspections
"""
import dataclasses
from enum import Enum
from typing import Tuple, List, Iterable, Dict


@dataclasses.dataclass(frozen=True)
class ColumnInfo:
    """
    A class we use to efficiently pass pandas/sklearn rows
    """
    fields: List[str]

    def get_index_of_column(self, column_name):
        """
        Get the values index for some column
        """
        if column_name in self.fields:
            return self.fields.index(column_name)
        return None

    def __eq__(self, other):
        return (isinstance(other, ColumnInfo) and
                self.fields == other.fields)


@dataclasses.dataclass(frozen=True)
class FunctionInfo:
    """
    Contains the function name and its path
    """
    module: str
    function_name: str


class OperatorType(Enum):
    """
    The different operator types in our DAG
    """
    DATA_SOURCE = "Data Source"
    MISSING_OP = "Encountered unsupported operation! Fallback: Data Source"
    SELECTION = "Selection"
    PROJECTION = "Projection"
    PROJECTION_MODIFY = "Projection (Modify)"
    TRANSFORMER = "Transformer"
    CONCATENATION = "Concatenation"
    ESTIMATOR = "Estimator"
    SCORE = "Score"
    TRAIN_DATA = "Train Data"
    TRAIN_LABELS = "Train Labels"
    TEST_DATA = "Test Data"
    TEST_LABELS = "Test Labels"
    JOIN = "Join"
    GROUP_BY_AGG = "Groupby and Aggregate"
    TRAIN_TEST_SPLIT = "Train Test Split"


@dataclasses.dataclass(frozen=True)
class OperatorContext:
    """
    Additional context for the inspection. Contains, most importantly, the operator type.
    """
    operator: OperatorType
    function_info: FunctionInfo or None


@dataclasses.dataclass(frozen=True)
class InspectionRowDataSource:
    """
    Wrapper class for the only operator without a parent: a Data Source
    """
    output: Tuple


@dataclasses.dataclass(frozen=True)
class InspectionInputDataSource:
    """
    Additional context for the inspection. Contains, most importantly, the operator type.
    """
    operator_context: OperatorContext
    output_columns: ColumnInfo
    row_iterator: Iterable[InspectionRowDataSource]
    non_data_function_args: Dict[str, any]


@dataclasses.dataclass(frozen=True)
class InspectionRowUnaryOperator:
    """
    Wrapper class for the operators with one parent like Selections and Projections
    """
    input: Tuple
    annotation: any
    output: Tuple


@dataclasses.dataclass(frozen=True)
class InspectionInputUnaryOperator:
    """
    Additional context for the inspection. Contains, most importantly, the operator type.
    """
    operator_context: OperatorContext
    input_columns: ColumnInfo
    output_columns: ColumnInfo
    row_iterator: Iterable[InspectionRowUnaryOperator]
    non_data_function_args: Dict[str, any]


@dataclasses.dataclass(frozen=True)
class InspectionRowNAryOperator:
    """
    Wrapper class for the operators with multiple parents like Concatenations
    """
    inputs: Tuple[Tuple]
    annotation: Tuple[any]
    output: Tuple


@dataclasses.dataclass(frozen=True)
class InspectionInputNAryOperator:
    """
    Additional context for the inspection. Contains, most importantly, the operator type.
    """
    operator_context: OperatorContext
    inputs_columns: List[ColumnInfo]
    output_columns: ColumnInfo
    row_iterator: Iterable[InspectionRowNAryOperator]
    non_data_function_args: Dict[str, any]


@dataclasses.dataclass(frozen=True)
class InspectionRowSinkOperator:
    """
    Wrapper class for operators like Estimators that only get fitted
    """
    input: Tuple[Tuple]
    annotation: any


@dataclasses.dataclass(frozen=True)
class InspectionInputSinkOperator:
    """
    Additional context for the inspection. Contains, most importantly, the operator type.
    """
    operator_context: OperatorContext
    inputs_columns: List[ColumnInfo]
    row_iterator: Iterable[InspectionRowSinkOperator]
    non_data_function_args: Dict[str, any]
