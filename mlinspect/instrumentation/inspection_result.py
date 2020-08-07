"""
Data class used as result of the PipelineExecutor
"""
import dataclasses
from typing import Dict, Tuple

import networkx

from mlinspect.instrumentation.analyzers.analyzer import Analyzer


@dataclasses.dataclass
class InspectionResult:
    """
    The class the PipelineExecutor returns
    """
    graph: networkx.DiGraph
    analyzer_to_annotations: Dict[Analyzer, Dict[Tuple[int, int], any]]
