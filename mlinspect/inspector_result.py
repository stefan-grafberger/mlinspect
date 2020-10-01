"""
Data class used as result of the PipelineExecutor
"""
import dataclasses
from typing import Dict, Tuple

import networkx

from mlinspect.checks.check import Check, CheckResult
from mlinspect.inspections.inspection import Inspection


@dataclasses.dataclass
class InspectorResult:
    """
    The class the PipelineExecutor returns
    """
    dag: networkx.DiGraph
    inspection_to_annotations: Dict[Inspection, Dict[Tuple[int, int], any]]
    check_to_check_results: Dict[Check, CheckResult]
