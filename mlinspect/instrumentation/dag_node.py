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


@dataclasses.dataclass
class DagNode:
    """
    A DAG Node
    """

    node_id: int
    operator_type: OperatorType
    lineno: int or None = None
    col_offset: int or None = None
    module: Tuple or None = None
    description: str or None = None

    def __hash__(self):
        return hash(self.node_id)
