"""
Data class used as result of the PipelineExecutor
"""
import dataclasses
from typing import Tuple, OrderedDict

import networkx

from mlinspect.inspections._inspection import Inspection


@dataclasses.dataclass
class InspectionResult:
    """
    The class the PipelineExecutor returns
    """
    dag: networkx.DiGraph
    inspection_to_annotations: OrderedDict[Inspection, OrderedDict[Tuple[int, int], any]]
