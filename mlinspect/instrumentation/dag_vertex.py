"""
The Vertices used in the WIR as nodes for the networkx.DiGraph
"""
from enum import Enum


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


class DagVertex:
    """
    A WIR Vertex
    """

    def __init__(self, node_id, operator_name, lineno=None, col_offset=None, module=None, description=None):
        # pylint: disable=too-many-arguments
        self.node_id = node_id
        self.operator_name = operator_name
        self.module = module
        self.lineno = lineno
        self.col_offset = col_offset
        self.description = description

    def __repr__(self):
        message = "DagVertex(node_id={}: operator_name='{}', module={}, lineno={}, col_offset={}, description='{}')" \
            .format(self.node_id, self.operator_name, self.module, self.lineno, self.col_offset, self.description)
        return message

    def __eq__(self, other):
        return isinstance(other, DagVertex) and \
               self.node_id == other.node_id and \
               self.operator_name == other.operator_name and \
               self.module == other.module and \
               self.lineno == other.lineno and \
               self.col_offset == other.col_offset and \
               self.description == other.description

    def __hash__(self):
        return hash(self.node_id)
