"""
The Nodes used in the WIR as nodes for the networkx.DiGraph
"""
import dataclasses
from typing import Tuple


@dataclasses.dataclass
class WirNode:
    """
    A WIR Node
    """
    node_id: int
    name: str
    operation: str
    lineno: int or None = None
    col_offset: int or None = None
    module: Tuple or None = None
    description: str or None = None

    def __hash__(self):
        return hash(self.node_id)
