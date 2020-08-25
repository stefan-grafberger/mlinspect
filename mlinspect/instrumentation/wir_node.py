"""
The Nodes used in the WIR as nodes for the networkx.DiGraph
"""
import dataclasses
from typing import Tuple

from ..instrumentation.dag_node import CodeReference


@dataclasses.dataclass
class WirNode:
    """
    A WIR Node
    """
    node_id: int
    name: str
    operation: str
    code_reference: CodeReference or None = None
    module: Tuple or None = None
    dag_operator_description: str or None = None

    def __hash__(self):
        return hash(self.node_id)
