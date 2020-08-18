"""
The Nodes used in the DAG as nodes for the networkx.DiGraph
"""
import dataclasses
from enum import Enum
from typing import Tuple


class OperatorType(Enum):
    """
    The different operator types in our DAG
    """
    DATA_SOURCE = "Data Source"
    SELECTION = "Selection"
    PROJECTION = "Projection"
    PROJECTION_MODIFY = "Projection (Modify)"
    TRANSFORMER = "Transformer"
    CONCATENATION = "Concatenation"
    ESTIMATOR = "Estimator"
    FIT = "Fit Transformers and Estimators"
    TRAIN_DATA = "Train Data"
    TRAIN_LABELS = "Train Labels"
    JOIN = "Join"
    GROUP_BY_AGG = "Groupby and Aggregate"
    TRAIN_TEST_SPLIT = "Train Test Split"


@dataclasses.dataclass(frozen=True)
class CodeReference:
    """
    Identifies a function call in the user pipeline code
    """
    lineno: int
    col_offset: int
    end_lineno: int
    end_col_offset: int


@dataclasses.dataclass
class DagNode:
    """
    A DAG Node
    """

    node_id: int
    operator_type: OperatorType
    code_reference: CodeReference or None = None
    module: Tuple or None = None
    description: str or None = None

    def __hash__(self):
        return hash(self.node_id)


@dataclasses.dataclass(frozen=True)
class DagNodeIdentifier:
    """
    Identifies a function call in the user pipeline code
    """
    operator_type: OperatorType
    code_reference: CodeReference
    description: str or None
