"""
Data class used as result of the PipelineExecutor
"""
import dataclasses
from typing import Dict

import networkx

from mlinspect.instrumentation._dag_node import DagNode
from mlinspect.inspections._inspection import Inspection


@dataclasses.dataclass
class InspectionResult:
    """
    The class the PipelineExecutor returns
    """
    dag: networkx.DiGraph
    dag_node_to_inspection_results: Dict[DagNode, Dict[Inspection, any]]
