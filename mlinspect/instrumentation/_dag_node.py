"""
The Nodes used in the DAG as nodes for the networkx.DiGraph
"""
import dataclasses
from enum import Enum
from typing import Tuple, List


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

    @property
    def short_value(self):
        rel_algebra_symbols = {
            OperatorType.SELECTION: "σ",
            OperatorType.PROJECTION: "π",
            OperatorType.PROJECTION_MODIFY: "π",
            OperatorType.JOIN: "⋈",
            OperatorType.GROUP_BY_AGG: "Γ",
            OperatorType.TRAIN_TEST_SPLIT: "◨",
            OperatorType.DATA_SOURCE: "☷",
            OperatorType.TRANSFORMER: "τ",
            OperatorType.ESTIMATOR: "f",
            OperatorType.FIT: "=",
            OperatorType.TRAIN_DATA: "χ",
            OperatorType.TRAIN_LABELS: "γ",
            OperatorType.CONCATENATION: "⧺"
        }
        if self in rel_algebra_symbols:
            return rel_algebra_symbols[self]
        return self.value[:1]


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
    columns: List[str] = None
    source_code: str or None = None

    def __hash__(self):
        return hash(self.node_id)

    def to_dict(self):
        return {
            'node_id': self.node_id,
            'operator_type': self.operator_type.value,
            'code_reference': {
                'lineno': self.code_reference.lineno,
                'col_offset': self.code_reference.col_offset,
                'end_lineno': self.code_reference.end_lineno,
                'end_col_offset': self.code_reference.end_col_offset,
            },
            'module': self.module,
            'description': self.description,
            'columns': self.columns,
            'source_code': self.source_code,
        }


@dataclasses.dataclass(frozen=True)
class DagNodeIdentifier:
    """
    Identifies a function call in the user pipeline code
    """
    operator_type: OperatorType
    code_reference: CodeReference
    description: str or None
