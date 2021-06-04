"""
Data class used as result of the PipelineExecutor
"""
import dataclasses
from typing import Dict

import networkx

from mlinspect.checks._check import Check, CheckResult
from mlinspect.inspections._inspection import Inspection


@dataclasses.dataclass
class InspectorResult:
    """
    The class the PipelineExecutor returns
    """
    dag: networkx.DiGraph
    dag_node_to_inspection_results: Dict[any, Dict[Inspection, any]]  # First any is DagNode
    check_to_check_results: Dict[Check, CheckResult]
