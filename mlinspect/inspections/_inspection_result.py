"""
Data class used as result of the PipelineExecutor
"""
import dataclasses

import networkx

from mlinspect.instrumentation._dag_node import DagNode
from mlinspect.inspections._inspection import Inspection


@dataclasses.dataclass
class InspectionResult:
    """
    The class the PipelineExecutor returns
    """
    dag: networkx.DiGraph
    inspection_to_annotations: dict[DagNode, dict[Inspection, any]]
