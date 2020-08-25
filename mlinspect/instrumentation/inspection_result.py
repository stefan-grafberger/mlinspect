"""
Data class used as result of the PipelineExecutor
"""
import dataclasses
from typing import Dict, Tuple

import networkx

from ..inspections.inspection import Inspection


@dataclasses.dataclass
class InspectionResult:
    """
    The class the PipelineExecutor returns
    """
    dag: networkx.DiGraph
    analyzer_to_annotations: Dict[Inspection, Dict[Tuple[int, int], any]]
